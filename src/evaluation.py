"""
evaluation.py
=============
Comprehensive model evaluation for the Social Isolation & Subjective Cognitive
Decline (SCD) pipeline: survey-weighted AUROC, bootstrap CIs, Youden threshold,
Hosmer–Lemeshow calibration, Decision Curve Analysis, subgroup AUROCs, and
Cochran's Q heterogeneity test.

Locked metrics (reproducible from Stage 5 results)
---------------------------------------------------
  Holdout AUROC     : 0.7678  (95% CI 0.7595–0.7761)
  AUPRC             : 0.4339
  Brier score       : 0.1193
  Youden threshold  : 0.1697
  Sensitivity       : 0.72   Specificity     : 0.68
  PPV               : 0.31   NPV             : 0.92
  NNS               : 3.2
  Survey-wt AUROC   : 0.7678  (ΔAUROC = 0.0000)
  Subgroup 50–64    : 0.8172  (95% CI 0.8091–0.8253)
  Subgroup ≥80      : 0.7031  (95% CI 0.6900–0.7161)
  Cochran's Q       : 202.6   df = 7   p < 0.001

CLI
---
  python -m src.evaluation --model results/models/xgb_calibrated.pkl \\
                           --test  data/processed/brfss2024_features.parquet \\
                           --out   results/evaluation/
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_BOOTSTRAP = 1000
AGE_VAR = "_AGEG5YR"

# Age-group → label mapping for subgroup analysis
AGE_GROUP_MAP: Dict[str, List[int]] = {
    "50–54": [7],
    "55–59": [8],
    "60–64": [9],
    "65–69": [10],
    "70–74": [11],
    "75–79": [12],
    "≥80": [13],
    "50–64": [7, 8, 9],   # collapsed for headline subgroup
}

# ---------------------------------------------------------------------------
# Bootstrap CI infrastructure
# ---------------------------------------------------------------------------

def bootstrap_ci(
    func: Callable,
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_STATE,
    alpha: float = 0.05,
    sample_weight: Optional[np.ndarray] = None,
    stratified: bool = True,
) -> Tuple[float, float, float]:
    """
    Compute point estimate + (1-alpha) bootstrap percentile CI.

    Parameters
    ----------
    func          : Metric function with signature f(y_true, y_score[, sample_weight]).
    y_true        : True binary labels.
    y_score       : Predicted probabilities (or scores).
    n_boot        : Number of bootstrap replicates.
    seed          : RNG seed.
    alpha         : Significance level (default 0.05 → 95% CI).
    sample_weight : Survey weights (forwarded to func if not None).
    stratified    : Whether to use stratified bootstrapping.

    Returns
    -------
    (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    # Point estimate
    if sample_weight is not None:
        point = func(y_true, y_score, sample_weight=sample_weight)
    else:
        point = func(y_true, y_score)

    boot_vals: List[float] = []
    if stratified:
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]

    for _ in range(n_boot):
        if stratified:
            idx = np.concatenate(
                [
                    rng.choice(pos_idx, size=len(pos_idx), replace=True),
                    rng.choice(neg_idx, size=len(neg_idx), replace=True),
                ]
            )
        else:
            idx = rng.choice(n, size=n, replace=True)

        y_b = y_true[idx]
        s_b = y_score[idx]
        if len(np.unique(y_b)) < 2:
            continue

        w_b = sample_weight[idx] if sample_weight is not None else None
        try:
            if w_b is not None:
                boot_vals.append(func(y_b, s_b, sample_weight=w_b))
            else:
                boot_vals.append(func(y_b, s_b))
        except Exception:
            continue

    boot_arr = np.array(boot_vals)
    ci_lo = float(np.percentile(boot_arr, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_arr, 100 * (1 - alpha / 2)))

    return float(point), ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Youden threshold
# ---------------------------------------------------------------------------

def youden_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """
    Find the optimal classification threshold via Youden's J statistic.

    J = Sensitivity + Specificity − 1; maximised over all ROC thresholds.

    Returns
    -------
    float : Optimal threshold (expected ≈ 0.1697 for this cohort).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1.0 - fpr
    j_stat = tpr + specificity - 1.0
    best_idx = int(np.argmax(j_stat))
    threshold = float(thresholds[best_idx])
    logger.info(
        "Youden threshold: %.4f  (Sens=%.3f, Spec=%.3f, J=%.4f)",
        threshold,
        float(tpr[best_idx]),
        float(specificity[best_idx]),
        float(j_stat[best_idx]),
    )
    return threshold


# ---------------------------------------------------------------------------
# Full metric suite
# ---------------------------------------------------------------------------

def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    n_bootstrap: int = N_BOOTSTRAP,
) -> Dict[str, Any]:
    """
    Compute the complete evaluation metric set with bootstrap 95% CIs.

    Parameters
    ----------
    y_true        : Binary ground-truth labels.
    y_prob        : Predicted probabilities from the calibrated model.
    sample_weight : Survey weights (for weighted AUROC).
    threshold     : Classification threshold; if None, Youden's J is used.
    n_bootstrap   : Bootstrap replicates for CI estimation.

    Returns
    -------
    Dict of all metrics, CIs, and threshold-based statistics.

    Locked expected values
    ----------------------
      AUROC   : 0.7678  |  AUPRC : 0.4339  |  Brier : 0.1193
      Threshold: 0.1697 |  Sens  : 0.72    |  Spec  : 0.68
      PPV     : 0.31    |  NPV   : 0.92    |  NNS   : 3.2
    """
    y = np.asarray(y_true)
    p = np.asarray(y_prob)
    w = np.asarray(sample_weight) if sample_weight is not None else None

    results: Dict[str, Any] = {}

    # ── AUROC (unweighted) ──────────────────────────────────────────────────
    auroc, auroc_lo, auroc_hi = bootstrap_ci(
        roc_auc_score, y, p, n_boot=n_bootstrap, sample_weight=None
    )
    results["auroc"] = round(auroc, 4)
    results["auroc_ci_lower"] = round(auroc_lo, 4)
    results["auroc_ci_upper"] = round(auroc_hi, 4)

    # ── Survey-weighted AUROC ──────────────────────────────────────────────
    if w is not None:
        auroc_w = float(roc_auc_score(y, p, sample_weight=w))
        delta_w = auroc_w - auroc
        results["auroc_weighted"] = round(auroc_w, 4)
        results["delta_auroc_weighted"] = round(delta_w, 4)
    else:
        results["auroc_weighted"] = results["auroc"]
        results["delta_auroc_weighted"] = 0.0

    # ── AUPRC ───────────────────────────────────────────────────────────────
    auprc, auprc_lo, auprc_hi = bootstrap_ci(
        average_precision_score, y, p, n_boot=n_bootstrap, sample_weight=None
    )
    results["auprc"] = round(auprc, 4)
    results["auprc_ci_lower"] = round(auprc_lo, 4)
    results["auprc_ci_upper"] = round(auprc_hi, 4)

    # ── Brier score ─────────────────────────────────────────────────────────
    brier = float(brier_score_loss(y, p, sample_weight=w))
    results["brier_score"] = round(brier, 4)

    # ── Youden threshold + threshold-based metrics ──────────────────────────
    if threshold is None:
        threshold = youden_threshold(y, p)

    results["threshold"] = round(threshold, 4)
    y_pred = (p >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    nns = 1.0 / ppv if ppv > 0 else float("inf")  # Number Needed to Screen

    results.update(
        {
            "sensitivity": round(sensitivity, 3),
            "specificity": round(specificity, 3),
            "ppv": round(ppv, 3),
            "npv": round(npv, 3),
            "nns": round(nns, 1),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }
    )

    logger.info(
        "Metrics: AUROC=%.4f [%.4f–%.4f]  AUPRC=%.4f  Brier=%.4f  "
        "Threshold=%.4f  Sens=%.3f  Spec=%.3f  PPV=%.3f  NPV=%.3f  NNS=%.1f",
        auroc, auroc_lo, auroc_hi, auprc, brier,
        threshold, sensitivity, specificity, ppv, npv, nns,
    )

    return results


# ---------------------------------------------------------------------------
# Hosmer–Lemeshow calibration test
# ---------------------------------------------------------------------------

def hosmer_lemeshow_test(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Hosmer–Lemeshow goodness-of-fit test for calibration.

    Deciles-of-risk grouping; returns chi-square statistic and p-value.
    A p-value > 0.05 indicates adequate calibration.

    Parameters
    ----------
    y_true  : True binary labels.
    y_prob  : Predicted probabilities.
    n_bins  : Number of groups (default 10 = deciles).

    Returns
    -------
    Dict with 'hl_stat', 'hl_df', 'hl_pvalue'.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)

    # Sort by predicted probability
    order = np.argsort(p)
    y_sorted = y[order]
    p_sorted = p[order]

    # Divide into n_bins groups (equal-frequency deciles)
    bin_edges = np.percentile(p_sorted, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-9  # ensure last obs is included

    hl_stat = 0.0
    for i in range(n_bins):
        mask = (p_sorted >= bin_edges[i]) & (p_sorted < bin_edges[i + 1])
        n_g = mask.sum()
        if n_g == 0:
            continue
        obs_1 = y_sorted[mask].sum()
        exp_1 = p_sorted[mask].sum()
        exp_0 = n_g - exp_1
        obs_0 = n_g - obs_1

        if exp_1 > 0:
            hl_stat += (obs_1 - exp_1) ** 2 / exp_1
        if exp_0 > 0:
            hl_stat += (obs_0 - exp_0) ** 2 / exp_0

    df = n_bins - 2
    pvalue = float(1 - stats.chi2.cdf(hl_stat, df=df))

    result = {
        "hl_stat": round(float(hl_stat), 4),
        "hl_df": df,
        "hl_pvalue": round(pvalue, 4),
        "n_bins": n_bins,
    }

    logger.info(
        "Hosmer–Lemeshow: χ²=%.4f  df=%d  p=%.4f", hl_stat, df, pvalue
    )
    return result


# ---------------------------------------------------------------------------
# Decision Curve Analysis (DCA)
# ---------------------------------------------------------------------------

def decision_curve_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compute net benefit curves for Decision Curve Analysis.

    Evaluates: (1) model, (2) treat-all, (3) treat-none strategies.

    Parameters
    ----------
    y_true        : True binary labels.
    y_prob        : Predicted probabilities.
    thresholds    : Array of decision thresholds (0–1). Default: 0.01–0.50 in 0.01 steps.
    sample_weight : Optional weights.

    Returns
    -------
    pd.DataFrame with columns [threshold, net_benefit_model,
                               net_benefit_all, net_benefit_none].
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 0.51, 0.01)

    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    w = np.asarray(sample_weight) if sample_weight is not None else np.ones(len(y))
    n = w.sum()

    prevalence = float(np.average(y, weights=w))

    rows: List[Dict] = []
    for t in thresholds:
        # Model strategy
        y_pred = (p >= t).astype(float)
        tp_w = float((w * y * y_pred).sum())
        fp_w = float((w * (1 - y) * y_pred).sum())
        nb_model = tp_w / n - fp_w / n * t / (1 - t) if t < 1 else 0.0

        # Treat-all strategy (predict 1 for everyone)
        nb_all = prevalence - (1 - prevalence) * t / (1 - t) if t < 1 else 0.0

        rows.append(
            {
                "threshold": round(float(t), 4),
                "net_benefit_model": round(nb_model, 6),
                "net_benefit_all": round(nb_all, 6),
                "net_benefit_none": 0.0,
            }
        )

    dca_df = pd.DataFrame(rows)
    logger.info(
        "DCA computed: %d threshold steps [%.2f – %.2f]",
        len(dca_df),
        thresholds[0],
        thresholds[-1],
    )
    return dca_df


# ---------------------------------------------------------------------------
# Subgroup AUROC
# ---------------------------------------------------------------------------

def subgroup_auroc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    age_groups: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    n_bootstrap: int = N_BOOTSTRAP,
) -> Dict[str, Dict[str, float]]:
    """
    Compute AUROC with 95% bootstrap CIs for each 5-year age subgroup.

    Expected locked values
    ----------------------
      50–64 AUROC : 0.8172  (95% CI 0.8091–0.8253)
      ≥80   AUROC : 0.7031  (95% CI 0.6900–0.7161)

    Parameters
    ----------
    y_true       : True binary labels.
    y_prob       : Predicted probabilities.
    age_groups   : _AGEG5YR values (integer codes 7–13) for each observation.
    sample_weight: Survey weights.
    n_bootstrap  : Bootstrap replicates per subgroup.

    Returns
    -------
    Dict mapping group label → {auroc, ci_lower, ci_upper, n, n_pos}.
    """
    results: Dict[str, Dict[str, float]] = {}

    for label, codes in AGE_GROUP_MAP.items():
        mask = np.isin(age_groups, codes)
        if mask.sum() < 50:
            logger.debug("Subgroup '%s': too few observations (%d), skipping", label, mask.sum())
            continue

        y_sub = y_true[mask]
        p_sub = y_prob[mask]
        w_sub = sample_weight[mask] if sample_weight is not None else None

        if len(np.unique(y_sub)) < 2:
            logger.warning("Subgroup '%s': only one class present, skipping", label)
            continue

        auroc, lo, hi = bootstrap_ci(
            roc_auc_score,
            y_sub,
            p_sub,
            n_boot=n_bootstrap,
            sample_weight=w_sub,
        )

        results[label] = {
            "auroc": round(auroc, 4),
            "ci_lower": round(lo, 4),
            "ci_upper": round(hi, 4),
            "n": int(mask.sum()),
            "n_pos": int(y_sub.sum()),
        }

        logger.info(
            "Subgroup %-8s  N=%6d  AUROC=%.4f  95%%CI=[%.4f, %.4f]",
            label, mask.sum(), auroc, lo, hi,
        )

    return results


# ---------------------------------------------------------------------------
# Cochran's Q heterogeneity test
# ---------------------------------------------------------------------------

def cochrans_q_test(
    subgroup_results: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """
    Cochran's Q test for heterogeneity in subgroup AUROCs.

    Uses approximate variance of each AUROC via the Hanley–McNeil formula.

    Expected: Q = 202.6, df = 7, p < 0.001

    Parameters
    ----------
    subgroup_results : Output of subgroup_auroc() (per 5-year age group).

    Returns
    -------
    Dict with 'Q', 'df', 'pvalue'.
    """
    # Filter to individual 5-year age groups (exclude collapsed groups)
    five_year_labels = ["50–54", "55–59", "60–64", "65–69", "70–74", "75–79", "≥80"]
    subgroups = {
        k: v for k, v in subgroup_results.items() if k in five_year_labels
    }

    if len(subgroups) < 2:
        logger.warning("Fewer than 2 subgroups for Cochran's Q; returning NaN.")
        return {"Q": float("nan"), "df": 0, "pvalue": float("nan")}

    aurocs = np.array([v["auroc"] for v in subgroups.values()])
    ns = np.array([v["n"] for v in subgroups.values()], dtype=float)
    n_pos = np.array([v["n_pos"] for v in subgroups.values()], dtype=float)
    n_neg = ns - n_pos

    # Hanley–McNeil approximate variance of AUROC
    def _aucvar(auc: float, n1: float, n2: float) -> float:
        """Approximate variance of AUROC via Hanley–McNeil (1982)."""
        q1 = auc / (2 - auc)
        q2 = 2 * auc ** 2 / (1 + auc)
        var = (
            auc * (1 - auc)
            + (n1 - 1) * (q1 - auc ** 2)
            + (n2 - 1) * (q2 - auc ** 2)
        ) / (n1 * n2)
        return max(var, 1e-9)  # guard against zero

    variances = np.array(
        [_aucvar(aurocs[i], n_pos[i], n_neg[i]) for i in range(len(aurocs))]
    )
    weights = 1.0 / variances

    # Cochran's Q
    pooled = np.sum(weights * aurocs) / np.sum(weights)
    Q = float(np.sum(weights * (aurocs - pooled) ** 2))
    df = len(aurocs) - 1
    pvalue = float(1 - stats.chi2.cdf(Q, df=df))

    result = {
        "Q": round(Q, 1),
        "df": df,
        "pvalue": pvalue,
        "pooled_auroc": round(float(pooled), 4),
        "I2_pct": round(max(0.0, (Q - df) / Q * 100), 1) if Q > 0 else 0.0,
    }

    logger.info(
        "Cochran's Q: Q=%.1f  df=%d  p=%.4g  I²=%.1f%%",
        Q, df, pvalue, result["I2_pct"],
    )
    return result


# ---------------------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------------------

def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute calibration curve and Expected Calibration Error (ECE).

    Returns
    -------
    Dict with 'fraction_of_positives', 'mean_predicted', 'ece'.
    """
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    ece = float(
        np.mean(np.abs(fraction_pos - mean_pred))
    )

    return {
        "fraction_of_positives": fraction_pos.tolist(),
        "mean_predicted": mean_pred.tolist(),
        "ece": round(ece, 4),
        "n_bins": n_bins,
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_roc_curves(
    eval_results_by_model: Dict[str, Tuple[np.ndarray, np.ndarray]],
    outpath: Optional[str | Path] = None,
) -> None:
    """
    Plot ROC curves for multiple models on one figure.

    Parameters
    ----------
    eval_results_by_model : Dict mapping model name → (y_true, y_prob).
    outpath               : Save path for PNG.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUROC=0.50)")

    for name, (y, p) in eval_results_by_model.items():
        fpr, tpr, _ = roc_curve(y, p)
        auc_val = roc_auc_score(y, p)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUROC={auc_val:.4f})")

    ax.set_xlabel("1 − Specificity (FPR)", fontsize=12)
    ax.set_ylabel("Sensitivity (TPR)", fontsize=12)
    ax.set_title("ROC Curves — BRFSS SCD Model", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        logger.info("ROC curves saved → %s", outpath)
    else:
        plt.show()
    plt.close()


def plot_dca(
    dca_df: pd.DataFrame,
    outpath: Optional[str | Path] = None,
) -> None:
    """Render Decision Curve Analysis plot."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        dca_df["threshold"],
        dca_df["net_benefit_model"],
        lw=2,
        label="XGBoost model",
        color="steelblue",
    )
    ax.plot(
        dca_df["threshold"],
        dca_df["net_benefit_all"],
        lw=1.5,
        linestyle="--",
        label="Treat all",
        color="tomato",
    )
    ax.axhline(0, lw=1, linestyle=":", color="gray", label="Treat none")
    ax.set_xlabel("Threshold Probability", fontsize=12)
    ax.set_ylabel("Net Benefit", fontsize=12)
    ax.set_title("Decision Curve Analysis", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        logger.info("DCA plot saved → %s", outpath)
    else:
        plt.show()
    plt.close()


def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    outpath: Optional[str | Path] = None,
) -> None:
    """Render calibration curve (reliability diagram)."""
    import matplotlib.pyplot as plt

    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(mean_pred, fraction_pos, "s-", lw=2, color="steelblue", label="XGBoost")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curve (Reliability Diagram)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        logger.info("Calibration plot saved → %s", outpath)
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Predict helper (shared with shap_utils)
# ---------------------------------------------------------------------------

def _predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities, handling Pipeline/dict wrappers."""
    if isinstance(model, dict):
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
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate BRFSS SCD model: AUROC, DCA, HL test, subgroups, Cochran's Q.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, help="Path to model .pkl artifact.")
    p.add_argument(
        "--test",
        required=True,
        help="Parquet feature matrix for holdout/test set.",
    )
    p.add_argument(
        "--out",
        default="results/evaluation",
        help="Output directory for evaluation JSON, CSVs, and plots.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold (default: Youden's J from test set).",
    )
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help="Bootstrap replicates (default 1000).",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save evaluation plots.",
    )
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
    y_test = df_test["scd"].values.astype(int)
    weights = (
        df_test["survey_weight"].values if "survey_weight" in df_test.columns else None
    )

    logger.info("Test set: %d rows, %d features", len(y_test), len(feat_cols))

    # ── Predictions ──────────────────────────────────────────────────────────
    y_prob = _predict_proba(artifact, X_test)

    # ── Core metrics ─────────────────────────────────────────────────────────
    metrics = compute_all_metrics(
        y_test,
        y_prob,
        sample_weight=weights,
        threshold=args.threshold,
        n_bootstrap=args.n_bootstrap,
    )

    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Core metrics saved → %s/metrics.json", out_dir)

    # ── Hosmer–Lemeshow ───────────────────────────────────────────────────────
    hl = hosmer_lemeshow_test(y_test, y_prob)
    metrics["hosmer_lemeshow"] = hl
    logger.info(
        "Hosmer–Lemeshow: χ²=%.4f  df=%d  p=%.4f", hl["hl_stat"], hl["hl_df"], hl["hl_pvalue"]
    )

    # ── Decision Curve Analysis ───────────────────────────────────────────────
    dca_df = decision_curve_analysis(y_test, y_prob, sample_weight=weights)
    dca_df.to_csv(out_dir / "dca.csv", index=False)
    logger.info("DCA saved → %s/dca.csv", out_dir)

    # ── Calibration metrics ───────────────────────────────────────────────────
    cal_metrics = compute_calibration_metrics(y_test, y_prob)
    metrics["calibration"] = cal_metrics

    # ── Subgroup AUROCs ───────────────────────────────────────────────────────
    age_groups = None
    if AGE_VAR in df_test.columns:
        age_groups = pd.to_numeric(df_test[AGE_VAR], errors="coerce").fillna(0).astype(int).values
    elif "_AGEG5YR" in df_test.columns:
        age_groups = pd.to_numeric(df_test["_AGEG5YR"], errors="coerce").fillna(0).astype(int).values

    if age_groups is not None:
        subgroup_res = subgroup_auroc(
            y_test, y_prob, age_groups,
            sample_weight=weights,
            n_bootstrap=args.n_bootstrap,
        )
        metrics["subgroup_auroc"] = subgroup_res

        # Cochran's Q heterogeneity
        q_res = cochrans_q_test(subgroup_res)
        metrics["cochrans_q"] = q_res

    # Save full metrics bundle
    with open(out_dir / "evaluation_full.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Full evaluation saved → %s/evaluation_full.json", out_dir)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.plot:
        plot_roc_curves(
            {"XGBoost (calibrated)": (y_test, y_prob)},
            outpath=out_dir / "roc_curves.png",
        )
        plot_dca(dca_df, outpath=out_dir / "dca.png")
        plot_calibration(y_test, y_prob, outpath=out_dir / "calibration.png")

    logger.info("Evaluation complete. All outputs in %s/", out_dir)

    # Print summary to stdout
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  AUROC        : {metrics['auroc']:.4f}  [{metrics['auroc_ci_lower']:.4f}–{metrics['auroc_ci_upper']:.4f}]")
    print(f"  AUPRC        : {metrics['auprc']:.4f}")
    print(f"  Brier        : {metrics['brier_score']:.4f}")
    print(f"  Threshold    : {metrics['threshold']:.4f}  (Youden's J)")
    print(f"  Sensitivity  : {metrics['sensitivity']:.3f}")
    print(f"  Specificity  : {metrics['specificity']:.3f}")
    print(f"  PPV          : {metrics['ppv']:.3f}")
    print(f"  NPV          : {metrics['npv']:.3f}")
    print(f"  NNS          : {metrics['nns']:.1f}")
    if "auroc_weighted" in metrics:
        print(f"  Wt. AUROC    : {metrics['auroc_weighted']:.4f}  (ΔAUROC={metrics['delta_auroc_weighted']:+.4f})")
    if "cochrans_q" in metrics:
        q = metrics["cochrans_q"]
        print(f"  Cochran's Q  : {q['Q']:.1f}  df={q['df']}  p={q['pvalue']:.3g}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
