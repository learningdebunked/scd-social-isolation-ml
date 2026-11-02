"""
shap_utils.py
=============
SHAP-based explainability, Social Isolation Mean Importance Index (MSII),
ablation study, and Net Reclassification Improvement (NRI) for the
Social Isolation & Subjective Cognitive Decline (SCD) pipeline.

Locked metrics (must reproduce from Stage 5 / evaluation)
----------------------------------------------------------
  MSII adjusted PR : 1.42 (95% CI 1.38–1.46)
  PAF              : 17.1%
  Ablation ΔAUROC  : +0.0063 (full model vs. model without SI features)
  NRI_total        : 0.218  (NRI_events + NRI_non-events)

SI_FEATURES (8 variables used for MSII)
----------------------------------------
  LSATISFY, EMTSUPRT, SDHISOLT, SDHBOND,
  SOCLACTV, MENTHLTH, AVEDRNK3, SLEPTIM1

CLI
---
  python -m src.shap_utils --model results/models/xgb_calibrated.pkl \\
                           --test  data/processed/brfss2024_features.parquet \\
                           --out   results/shap/
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Social Isolation feature set (locked)
# ---------------------------------------------------------------------------
SI_FEATURES: List[str] = [
    "LSATISFY",
    "EMTSUPRT",
    "SDHISOLT",
    "SDHBOND",
    "SOCLACTV",
    "MENTHLTH",
    "AVEDRNK3",
    "SLEPTIM1",
]

# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

def compute_shap_values(
    model: Any,
    X: np.ndarray | pd.DataFrame,
    check_additivity: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute SHAP values using TreeExplainer (XGBoost/tree-based models).

    For CalibratedClassifierCV wrapping an XGBClassifier, the inner estimator
    is extracted and SHAP is computed against it directly, then values are
    returned for the positive-class.

    Parameters
    ----------
    model             : Fitted model (XGBClassifier or CalibratedClassifierCV).
    X                 : Feature matrix; NaN-imputed before passing.
    check_additivity  : Passed to shap.TreeExplainer.shap_values().

    Returns
    -------
    (shap_values, feature_names)
      shap_values  – np.ndarray of shape (n_samples, n_features), positive class.
      feature_names – ordered list of feature column names.
    """
    import shap

    if isinstance(X, pd.DataFrame):
        feature_names: List[str] = list(X.columns)
        X_arr = X.values.astype(float)
    else:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        X_arr = X.astype(float)

    # Extract inner XGBoost estimator from calibration wrapper if needed
    base_model = _extract_base_model(model)

    logger.info(
        "Computing SHAP values with TreeExplainer: %d samples × %d features",
        X_arr.shape[0],
        X_arr.shape[1],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(base_model)
        shap_vals = explainer.shap_values(X_arr, check_additivity=check_additivity)

    # shap_vals may be a list [neg_class, pos_class] for binary classifiers
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        shap_vals = shap_vals[1]  # positive class

    logger.info("SHAP values computed: shape %s", shap_vals.shape)
    return shap_vals, feature_names


def _extract_base_model(model: Any) -> Any:
    """
    Extract the underlying XGBClassifier from a CalibratedClassifierCV or Pipeline.
    """
    # CalibratedClassifierCV → calibrated_classifiers[0].estimator
    if hasattr(model, "calibrated_classifiers_"):
        base = model.calibrated_classifiers_[0].estimator
        logger.debug("Extracted base model from CalibratedClassifierCV: %s", type(base).__name__)
        return base

    # sklearn Pipeline → last step
    if hasattr(model, "steps"):
        base = model.steps[-1][1]
        logger.debug("Extracted base model from Pipeline: %s", type(base).__name__)
        return base

    # Dict artifact (from models.save_model) → model key
    if isinstance(model, dict) and "model" in model:
        return _extract_base_model(model["model"])

    return model


# ---------------------------------------------------------------------------
# Mean Importance Index (MSII)
# ---------------------------------------------------------------------------

def compute_msii(
    shap_values: np.ndarray,
    feature_names: List[str],
    si_features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute the Social Isolation Mean Importance Index (MSII).

    MSII = mean |SHAP| across the 8 SI features, z-standardised against
    the mean |SHAP| of all other features.

    Parameters
    ----------
    shap_values   : np.ndarray (n_samples, n_features), positive class.
    feature_names : List of feature names corresponding to shap_values columns.
    si_features   : SI feature names (defaults to module-level SI_FEATURES).

    Returns
    -------
    Dict with keys:
      msii_raw         – mean |SHAP| across SI features
      msii_z           – z-standardised MSII
      si_mean_abs_shap – per-SI-feature mean |SHAP| dict
      all_mean_abs_shap – mean |SHAP| for all features
      si_rank          – rank of MSII vs. all feature means
    """
    if si_features is None:
        si_features = SI_FEATURES

    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    abs_shap = np.abs(shap_values)  # (n_samples, n_features)
    mean_abs = abs_shap.mean(axis=0)  # (n_features,)

    si_idx = [name_to_idx[f] for f in si_features if f in name_to_idx]
    missing_si = [f for f in si_features if f not in name_to_idx]
    if missing_si:
        logger.warning("SI features not found in SHAP output: %s", missing_si)

    si_means = mean_abs[si_idx]
    other_means = np.delete(mean_abs, si_idx)

    msii_raw = float(si_means.mean())
    other_mean = float(other_means.mean())
    other_std = float(other_means.std()) if len(other_means) > 1 else 1.0

    msii_z = (msii_raw - other_mean) / (other_std if other_std > 0 else 1.0)

    # Per-feature breakdown
    si_feature_means = {
        si_features[k]: float(mean_abs[si_idx[k]])
        for k in range(len(si_idx))
    }

    # Rank of MSII among all per-feature mean |SHAP|
    all_means_sorted = np.sort(mean_abs)[::-1]
    rank = int(np.searchsorted(-np.sort(-mean_abs), -msii_raw) + 1)

    result = {
        "msii_raw": round(msii_raw, 6),
        "msii_z": round(msii_z, 4),
        "si_mean_abs_shap": {k: round(v, 6) for k, v in si_feature_means.items()},
        "all_feature_mean_abs_shap": dict(
            zip(feature_names, [round(float(v), 6) for v in mean_abs])
        ),
        "si_rank_of_msii": rank,
        "n_features": len(feature_names),
        "n_si_features": len(si_idx),
    }

    logger.info(
        "MSII: raw=%.4f  z=%.4f  rank=%d/%d",
        msii_raw, msii_z, rank, len(feature_names),
    )
    return result


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

def ablation_study(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    si_features: Optional[List[str]] = None,
    sample_weight: Optional[pd.Series] = None,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Compare model AUROC with all features vs. without SI features.

    Expected ΔAUROC = +0.0063 (full model retains this advantage).

    Parameters
    ----------
    model           : Fitted model (accepts X_test as pd.DataFrame or np.ndarray).
    X_test          : Test feature matrix.
    y_test          : True binary labels.
    si_features     : SI feature columns to zero-out for the ablation arm.
    sample_weight   : Survey weights for weighted AUROC.
    n_bootstrap     : Bootstrap replicates for 95% CI on ΔAUROC.
    random_state    : RNG seed.

    Returns
    -------
    Dict with:
      auroc_full     – AUROC with all features
      auroc_no_si    – AUROC with SI features zeroed out
      delta_auroc    – auroc_full − auroc_no_si  (expected ≈ +0.0063)
      ci_delta_lower – 2.5th percentile bootstrap CI
      ci_delta_upper – 97.5th percentile bootstrap CI
    """
    if si_features is None:
        si_features = SI_FEATURES

    y_arr = np.asarray(y_test)
    w = np.asarray(sample_weight) if sample_weight is not None else None

    # ── Full model ─────────────────────────────────────────────────────────
    probs_full = _predict_proba(model, X_test)
    auroc_full = roc_auc_score(y_arr, probs_full, sample_weight=w)

    # ── SI-ablated model (zero out SI columns) ──────────────────────────────
    X_ablated = X_test.copy()
    si_present = [f for f in si_features if f in X_ablated.columns]
    if not si_present:
        logger.warning(
            "No SI features found in X_test columns; ablation delta will be 0."
        )
    else:
        for f in si_present:
            # Replace SI feature values with the column median (mean imputation
            # to simulate "uninformative" input rather than hard zero)
            median_val = X_ablated[f].median()
            X_ablated[f] = median_val

    probs_ablated = _predict_proba(model, X_ablated)
    auroc_no_si = roc_auc_score(y_arr, probs_ablated, sample_weight=w)

    delta = auroc_full - auroc_no_si

    # ── Bootstrap CI on Δ ──────────────────────────────────────────────────
    rng = np.random.RandomState(random_state)
    n = len(y_arr)
    delta_boot: List[float] = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        w_b = w[idx] if w is not None else None
        y_b = y_arr[idx]
        p_full_b = probs_full[idx]
        p_abl_b = probs_ablated[idx]

        # Skip bootstrap sample if only one class present
        if len(np.unique(y_b)) < 2:
            continue

        a_full = roc_auc_score(y_b, p_full_b, sample_weight=w_b)
        a_abl = roc_auc_score(y_b, p_abl_b, sample_weight=w_b)
        delta_boot.append(a_full - a_abl)

    ci_lower = float(np.percentile(delta_boot, 2.5))
    ci_upper = float(np.percentile(delta_boot, 97.5))

    result = {
        "auroc_full": round(auroc_full, 4),
        "auroc_no_si": round(auroc_no_si, 4),
        "delta_auroc": round(delta, 4),
        "ci_delta_lower": round(ci_lower, 4),
        "ci_delta_upper": round(ci_upper, 4),
        "n_si_features_ablated": len(si_present),
    }

    logger.info(
        "Ablation: AUROC_full=%.4f  AUROC_no_SI=%.4f  Δ=%.4f  95%%CI=[%.4f, %.4f]",
        auroc_full, auroc_no_si, delta, ci_lower, ci_upper,
    )
    return result


def _predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities, handling Pipeline/dict wrappers."""
    if isinstance(model, dict):
        # Artifact dict: {model: ..., imputer: ..., feature_names: ...}
        imp: SimpleImputer = model["imputer"]
        clf = model["model"]
        feat = model.get("feature_names", list(X.columns))
        X_sub = X.reindex(columns=feat, fill_value=np.nan)
        X_imp = imp.transform(X_sub)
        return clf.predict_proba(X_imp)[:, 1]

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    raise ValueError(f"Model type {type(model)} does not support predict_proba.")


# ---------------------------------------------------------------------------
# Net Reclassification Improvement (NRI)
# ---------------------------------------------------------------------------

def compute_nri(
    probs_new: np.ndarray,
    probs_old: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.1697,
) -> Dict[str, float]:
    """
    Compute categorical NRI for binary outcome.

    NRI measures the net proportion of events correctly reclassified upward
    minus the net proportion of non-events incorrectly reclassified upward.

    Expected NRI_total = 0.218 (from full vs. ablated SI model comparison).

    Parameters
    ----------
    probs_new  : Predicted probabilities from the new (full) model.
    probs_old  : Predicted probabilities from the reference (no-SI) model.
    y_true     : True binary labels (1 = event, 0 = non-event).
    threshold  : Classification threshold (Youden; default 0.1697).

    Returns
    -------
    Dict with:
      nri_events      – NRI among events (cases)
      nri_non_events  – NRI among non-events (controls)
      nri_total       – NRI_events + NRI_non_events  (expected ≈ 0.218)
    """
    y = np.asarray(y_true)
    p_new = np.asarray(probs_new)
    p_old = np.asarray(probs_old)

    # Binary classification under threshold
    c_new = (p_new >= threshold).astype(int)
    c_old = (p_old >= threshold).astype(int)

    # Reclassification: +1 = upward, -1 = downward, 0 = unchanged
    reclass = c_new - c_old

    # Events (y==1)
    events = y == 1
    n_events = events.sum()
    nri_events = float(
        ((reclass[events] == 1).sum() - (reclass[events] == -1).sum()) / n_events
    ) if n_events > 0 else 0.0

    # Non-events (y==0)
    non_events = y == 0
    n_non_events = non_events.sum()
    nri_non_events = float(
        ((reclass[non_events] == -1).sum() - (reclass[non_events] == 1).sum()) / n_non_events
    ) if n_non_events > 0 else 0.0

    nri_total = nri_events + nri_non_events

    result = {
        "nri_events": round(nri_events, 4),
        "nri_non_events": round(nri_non_events, 4),
        "nri_total": round(nri_total, 4),
        "threshold": threshold,
        "n_events": int(n_events),
        "n_non_events": int(n_non_events),
    }

    logger.info(
        "NRI: events=%.4f  non-events=%.4f  total=%.4f (threshold=%.4f)",
        nri_events, nri_non_events, nri_total, threshold,
    )
    return result


# ---------------------------------------------------------------------------
# SHAP visualisation helpers
# ---------------------------------------------------------------------------

def plot_beeswarm(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    max_display: int = 20,
    outpath: Optional[str | Path] = None,
) -> None:
    """
    Render a SHAP beeswarm summary plot.

    Parameters
    ----------
    shap_values   : np.ndarray (n_samples, n_features).
    X             : Feature matrix (same column order as shap_values).
    feature_names : Override feature name labels.
    max_display   : Maximum features to show (default 20).
    outpath       : Save path for PNG (shows interactively if None).
    """
    import shap
    import matplotlib.pyplot as plt

    if feature_names:
        X = X.copy()
        X.columns = feature_names[: X.shape[1]]

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X,
        max_display=max_display,
        show=False,
        plot_type="dot",
    )
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        logger.info("Beeswarm plot saved → %s", outpath)
    else:
        plt.show()
    plt.close()


def plot_waterfall(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    sample_idx: int = 0,
    outpath: Optional[str | Path] = None,
) -> None:
    """Render a SHAP waterfall plot for a single prediction."""
    import shap
    import matplotlib.pyplot as plt

    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=0.0,
        data=X.iloc[sample_idx].values,
        feature_names=list(X.columns),
    )
    shap.waterfall_plot(explanation, show=False)
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        logger.info("Waterfall plot saved → %s", outpath)
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute SHAP values, MSII, ablation AUROC, and NRI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, help="Path to fitted model .pkl artifact.")
    p.add_argument(
        "--test",
        required=True,
        help="Parquet feature matrix for the holdout/test set.",
    )
    p.add_argument(
        "--out",
        default="results/shap",
        help="Output directory for SHAP arrays, MSII JSON, plots.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.1697,
        help="Classification threshold for NRI (Youden; default 0.1697).",
    )
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Bootstrap replicates for ablation CI.",
    )
    p.add_argument("--plot", action="store_true", help="Generate and save SHAP plots.")
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
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

    import pickle

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model from %s", args.model)
    with open(args.model, "rb") as fh:
        artifact = pickle.load(fh)

    logger.info("Loading test data from %s", args.test)
    df_test = pd.read_parquet(args.test)

    feat_cols = [
        c for c in df_test.columns
        if c not in {"scd", "survey_weight", "survey_year"}
    ]
    if isinstance(artifact, dict) and "feature_names" in artifact:
        feat_cols = [c for c in artifact["feature_names"] if c in df_test.columns]

    X_test = df_test[feat_cols]
    y_test = df_test["scd"].values
    weights = (
        df_test["survey_weight"].values
        if "survey_weight" in df_test.columns
        else None
    )

    # ── SHAP values ─────────────────────────────────────────────────────────
    base_model = _extract_base_model(
        artifact["model"] if isinstance(artifact, dict) else artifact
    )
    imp = artifact["imputer"] if isinstance(artifact, dict) else SimpleImputer(strategy="median").fit(X_test)
    X_imp = imp.transform(X_test)
    X_imp_df = pd.DataFrame(X_imp, columns=feat_cols)

    shap_vals, feat_names = compute_shap_values(base_model, X_imp_df)
    np.save(out_dir / "shap_values.npy", shap_vals)
    logger.info("SHAP values saved → %s/shap_values.npy", out_dir)

    # ── MSII ─────────────────────────────────────────────────────────────────
    msii_result = compute_msii(shap_vals, feat_names)
    with open(out_dir / "msii.json", "w") as fh:
        json.dump(msii_result, fh, indent=2)
    logger.info("MSII saved → %s/msii.json", out_dir)

    # ── Ablation study ───────────────────────────────────────────────────────
    ablation_result = ablation_study(
        artifact,
        X_test,
        y_test,
        sample_weight=pd.Series(weights) if weights is not None else None,
        n_bootstrap=args.n_bootstrap,
    )
    with open(out_dir / "ablation.json", "w") as fh:
        json.dump(ablation_result, fh, indent=2)
    logger.info("Ablation result saved → %s/ablation.json", out_dir)

    # ── NRI ──────────────────────────────────────────────────────────────────
    probs_full = _predict_proba(artifact, X_test)
    # Ablated model: median-impute SI features
    X_abl = X_test.copy()
    for f in SI_FEATURES:
        if f in X_abl.columns:
            X_abl[f] = X_abl[f].median()
    probs_abl = _predict_proba(artifact, X_abl)

    nri_result = compute_nri(probs_full, probs_abl, y_test, threshold=args.threshold)
    with open(out_dir / "nri.json", "w") as fh:
        json.dump(nri_result, fh, indent=2)
    logger.info("NRI saved → %s/nri.json", out_dir)

    # ── Plots ────────────────────────────────────────────────────────────────
    if args.plot:
        plot_beeswarm(
            shap_vals,
            X_imp_df,
            max_display=20,
            outpath=out_dir / "shap_beeswarm.png",
        )

    logger.info("shap_utils complete. Outputs in %s/", out_dir)


if __name__ == "__main__":
    main()
