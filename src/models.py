"""
models.py
=========
Model training, hyperparameter optimisation, calibration, and persistence
for the Social Isolation & Subjective Cognitive Decline (SCD) pipeline.

Architecture
------------
  Primary model : XGBoost 2.0.3 tuned with Optuna 3.5.0 Bayesian optimisation
                  (100 trials, 5-fold stratified CV, AUROC objective).
  Calibration   : Platt scaling via CalibratedClassifierCV(method='sigmoid').
  Baselines     : Logistic Regression, Decision Tree, Random Forest,
                  Gradient Boosting, Naive Bayes.

Locked hyperparameter search space (from Stage 4 engineering spec)
------------------------------------------------------------------
  n_estimators      : 400–1200
  max_depth         : 3–9
  learning_rate     : 1e-3 – 0.3  (log-uniform)
  subsample         : 0.5 – 1.0
  colsample_bytree  : 0.5 – 1.0
  min_child_weight  : 1 – 10
  gamma             : 0 – 5
  reg_alpha         : 1e-5 – 10.0 (log-uniform)
  reg_lambda        : 1e-5 – 10.0 (log-uniform)
  scale_pos_weight  : 1 – 8  (handles class imbalance; ~17% SCD)

CLI
---
  python -m src.models --train data/processed/brfss2023_features.parquet \\
                       --model-out results/models/xgb_calibrated.pkl \\
                       --baselines
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_OPTUNA_TRIALS = 100
CV_FOLDS = 5
OPTUNA_METRIC = "roc_auc"
FEATURE_COLS_EXCLUDE = {"scd", "survey_weight", "survey_year"}

# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def _split_X_y_w(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a feature-matrix DataFrame into (X, y, weights).

    Parameters
    ----------
    df           : Output of features.build_features() with 'scd' column attached.
    feature_cols : Explicit list of feature columns; if None, all columns except
                   known meta-columns are used.

    Returns
    -------
    (X, y, weights)  — all with matching index.
    """
    if "scd" not in df.columns:
        raise KeyError("Column 'scd' not found — is this the feature matrix with outcome?")

    y = df["scd"].astype(int)
    w = df["survey_weight"] if "survey_weight" in df.columns else pd.Series(
        np.ones(len(df)), index=df.index
    )

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in FEATURE_COLS_EXCLUDE]

    X = df[feature_cols].copy()
    return X, y, w


def _make_impute_pipeline(model: Any) -> Pipeline:
    """Wrap *model* in a median-impute → StandardScaler → model Pipeline."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", model),
        ]
    )


# ---------------------------------------------------------------------------
# XGBoost + Optuna tuning
# ---------------------------------------------------------------------------

def _xgb_objective(trial, X_imp: np.ndarray, y: np.ndarray, cv: StratifiedKFold):
    """Optuna objective: 5-fold stratified CV AUROC for XGBoost."""
    import xgboost as xgb

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1200),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 8.0),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "eval_metric": "auc",
        "use_label_encoder": False,
    }

    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(
        model, X_imp, y, cv=cv, scoring="roc_auc", n_jobs=-1
    )
    return scores.mean()


def tune_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = N_OPTUNA_TRIALS,
    cv_folds: int = CV_FOLDS,
    timeout_seconds: Optional[int] = None,
    show_progress: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Run Bayesian hyperparameter optimisation for XGBoost via Optuna.

    Parameters
    ----------
    X              : Feature matrix (NaN allowed; imputed internally).
    y              : Binary outcome series.
    n_trials       : Number of Optuna trials (default 100).
    cv_folds       : Stratified CV folds (default 5).
    timeout_seconds: Wall-clock limit in seconds (None = no limit).
    show_progress  : Show Optuna progress bar.

    Returns
    -------
    (best_model, best_params)
      best_model  – Fitted XGBClassifier with best hyperparameters.
      best_params – Dict of best hyperparameters with CV AUROC.
    """
    import optuna
    import xgboost as xgb

    optuna.logging.set_verbosity(
        optuna.logging.INFO if show_progress else optuna.logging.WARNING
    )

    # Impute NaN before passing to Optuna (XGBoost handles NaN natively,
    # but StratifiedKFold CV scoring is cleaner with imputed data)
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    y_arr = y.values

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )

    logger.info("Starting Optuna: %d trials, %d-fold CV", n_trials, cv_folds)
    study.optimize(
        lambda trial: _xgb_objective(trial, X_imp, y_arr, cv),
        n_trials=n_trials,
        timeout=timeout_seconds,
        show_progress_bar=show_progress,
        n_jobs=1,  # parallel CV is done inside; outer must be 1 for reproducibility
    )

    best = study.best_trial
    logger.info(
        "Best trial #%d: CV AUROC=%.4f | %s",
        best.number,
        best.value,
        best.params,
    )

    best_params = dict(best.params)
    best_params.update(
        {
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "eval_metric": "auc",
            "use_label_encoder": False,
        }
    )

    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(X_imp, y_arr)

    return best_model, {**best_params, "cv_auroc": best.value}


# ---------------------------------------------------------------------------
# Platt calibration
# ---------------------------------------------------------------------------

def calibrate_model(
    model: Any,
    X_cal: np.ndarray | pd.DataFrame,
    y_cal: pd.Series | np.ndarray,
    cv: int = 5,
) -> CalibratedClassifierCV:
    """
    Apply Platt scaling (sigmoid calibration) to a fitted classifier.

    Parameters
    ----------
    model : Already-fitted base classifier (e.g. XGBClassifier).
    X_cal : Calibration set features.
    y_cal : Calibration set labels.
    cv    : 'prefit' to use the already-fitted model; int for cross-val.
            Defaults to 5-fold (re-trains base model on each fold).

    Returns
    -------
    CalibratedClassifierCV fitted on (X_cal, y_cal).

    Notes
    -----
    The pipeline uses cv='prefit' with a dedicated calibration split drawn
    from the dev set (after XGBoost is trained on the training portion).
    """
    if isinstance(X_cal, pd.DataFrame):
        X_cal = SimpleImputer(strategy="median").fit_transform(X_cal)

    calibrated = CalibratedClassifierCV(estimator=model, method="sigmoid", cv="prefit")
    calibrated.fit(X_cal, y_cal)
    logger.info("Platt calibration complete (method=sigmoid, cv=prefit)")
    return calibrated


# ---------------------------------------------------------------------------
# Full XGBoost training pipeline
# ---------------------------------------------------------------------------

def train_xgboost_pipeline(
    df_train: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    cal_fraction: float = 0.15,
    n_trials: int = N_OPTUNA_TRIALS,
) -> Tuple[CalibratedClassifierCV, SimpleImputer, List[str], Dict[str, Any]]:
    """
    Full XGBoost training: impute → tune → calibrate.

    Parameters
    ----------
    df_train      : Training DataFrame with 'scd' and 'survey_weight' columns.
    feature_cols  : Feature columns (auto-detected if None).
    cal_fraction  : Fraction of training data held out for Platt calibration.
    n_trials      : Optuna trials.

    Returns
    -------
    (calibrated_model, imputer, feature_names, metadata)
    """
    from sklearn.model_selection import train_test_split

    X, y, w = _split_X_y_w(df_train, feature_cols)
    feature_names = list(X.columns)

    # Split off calibration set (stratified)
    X_tune, X_cal, y_tune, y_cal = train_test_split(
        X,
        y,
        test_size=cal_fraction,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    logger.info(
        "Tuning set: %d rows | Calibration set: %d rows", len(X_tune), len(X_cal)
    )

    # Impute
    imputer = SimpleImputer(strategy="median")
    X_tune_imp = imputer.fit_transform(X_tune)
    X_cal_imp = imputer.transform(X_cal)

    # Tune
    best_model, best_params = tune_xgboost(
        pd.DataFrame(X_tune_imp, columns=feature_names),
        y_tune.reset_index(drop=True),
        n_trials=n_trials,
    )

    # Calibrate
    calibrated = calibrate_model(best_model, X_cal_imp, y_cal.values)

    metadata = {
        "feature_names": feature_names,
        "n_train": len(X_tune),
        "n_cal": len(X_cal),
        "best_params": best_params,
    }

    return calibrated, imputer, feature_names, metadata


# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------

def train_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = CV_FOLDS,
) -> Dict[str, Pipeline]:
    """
    Train all baseline classifiers and report cross-validated AUROCs.

    Baselines
    ---------
    LogReg          : L2 logistic regression (C=1.0)
    DecisionTree    : max_depth=5
    RandomForest    : 500 trees
    GradientBoosting: sklearn GBM (100 estimators)
    NaiveBayes      : GaussianNB

    Parameters
    ----------
    X_train  : Imputed training feature matrix (no NaNs).
    y_train  : Binary labels.
    cv_folds : CV folds for AUROC reporting (default 5).

    Returns
    -------
    Dict mapping model name → fitted Pipeline (impute → scale → clf).
    """
    baselines: Dict[str, Any] = {
        "LogReg": LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=5, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=RANDOM_STATE
        ),
        "NaiveBayes": GaussianNB(),
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    fitted: Dict[str, Pipeline] = {}

    for name, clf in baselines.items():
        pipe = _make_impute_pipeline(clf)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        logger.info(
            "Baseline %-20s  CV AUROC = %.4f ± %.4f",
            name,
            scores.mean(),
            scores.std(),
        )
        pipe.fit(X_train, y_train)
        fitted[name] = pipe

    return fitted


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(
    model: Any,
    path: str | Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a model (and optional metadata) to disk with pickle/joblib.

    Saves:
      <path>.pkl  – the model object
      <path>.json – metadata dict (if provided)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pkl_path = path.with_suffix(".pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Model saved → %s (%.1f KB)", pkl_path, pkl_path.stat().st_size / 1024)

    if metadata is not None:
        meta_path = path.with_suffix(".json")
        # Convert numpy types to Python native for JSON serialisation
        def _jsonify(obj: Any) -> Any:
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        clean_meta = {k: _jsonify(v) for k, v in metadata.items()}
        with open(meta_path, "w") as fh:
            json.dump(clean_meta, fh, indent=2)
        logger.info("Metadata saved → %s", meta_path)


def load_model(path: str | Path) -> Any:
    """
    Load a model previously saved by save_model().

    Parameters
    ----------
    path : Path to the .pkl file (with or without extension).

    Returns
    -------
    Loaded model object.
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".pkl")
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    with open(path, "rb") as fh:
        model = pickle.load(fh)
    logger.info("Model loaded from %s", path)
    return model


def load_model_metadata(path: str | Path) -> Optional[Dict[str, Any]]:
    """Load metadata JSON saved alongside a model."""
    path = Path(path).with_suffix(".json")
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train XGBoost + calibration and baseline models for BRFSS SCD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--train",
        required=True,
        help="Path to feature-matrix Parquet (from features.py).",
    )
    p.add_argument(
        "--model-out",
        default="results/models/xgb_calibrated",
        help="Stem path for saving model (.pkl) and metadata (.json).",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=N_OPTUNA_TRIALS,
        help="Number of Optuna trials.",
    )
    p.add_argument(
        "--cal-fraction",
        type=float,
        default=0.15,
        help="Fraction of training data for Platt calibration split.",
    )
    p.add_argument(
        "--baselines",
        action="store_true",
        help="Train and save baseline classifiers.",
    )
    p.add_argument(
        "--baselines-out",
        default="results/models/baselines",
        help="Directory to save baseline model pickles.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    )

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Loading training data from %s", args.train)
    df = pd.read_parquet(args.train)
    logger.info("Loaded %d rows × %d cols", *df.shape)

    # ── Train primary XGBoost model ─────────────────────────────────────────
    cal_model, imputer, feature_names, meta = train_xgboost_pipeline(
        df,
        n_trials=args.n_trials,
        cal_fraction=args.cal_fraction,
    )

    artifact = {"model": cal_model, "imputer": imputer, "feature_names": feature_names}
    save_model(artifact, args.model_out, metadata=meta)

    # ── Baseline models ─────────────────────────────────────────────────────
    if args.baselines:
        X, y, _ = _split_X_y_w(df, feature_names)
        imp = SimpleImputer(strategy="median")
        X_imp = imp.fit_transform(X)

        baselines = train_baselines(X_imp, y.values)
        bl_dir = Path(args.baselines_out)
        bl_dir.mkdir(parents=True, exist_ok=True)
        for name, pipe in baselines.items():
            save_model(pipe, bl_dir / name)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
