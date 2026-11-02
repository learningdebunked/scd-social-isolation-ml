"""
features.py
===========
Feature engineering pipeline for the Social Isolation & Subjective Cognitive
Decline (SCD) prediction model.

Covers 7 domains (117 candidate variables) and applies a LASSO pre-screen
to select the final modelling feature set.

Locked social-isolation features (SI_FEATURES, 8 variables)
------------------------------------------------------------
  LSATISFY   – Life satisfaction (proxy for isolation affect)
  EMTSUPRT   – How often emotional support needed (BRFSS social context)
  SDHISOLT   – Social isolation (SDOH module, direct)
  SDHBOND    – Sense of belonging
  SOCLACTV   – Socially active in last 30 days
  MENTHLTH   – Days of poor mental health (last 30 days)
  AVEDRNK3   – Average drinks per day (social behavior proxy)
  SLEPTIM1   – Hours of sleep (isolation-linked health behavior)

CLI
---
  python -m src.features --input data/processed/brfss2023_cohort.parquet \\
                         --output data/processed/brfss2023_features.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain feature lists  (BRFSS variable names, upper-case)
# ---------------------------------------------------------------------------

# ── Social Isolation domain (primary exposure, 8 features) ─────────────────
SI_FEATURES: List[str] = [
    "LSATISFY",   # Life satisfaction (1=Very satisfied … 4=Very dissatisfied)
    "EMTSUPRT",   # Emotional support received (1=Always … 5=Never)
    "SDHISOLT",   # Feels socially isolated (1=Always … 5=Never)
    "SDHBOND",    # Sense of belonging to community (1=Always … 5=Never)
    "SOCLACTV",   # Socially active (1=Yes, 2=No)
    "MENTHLTH",   # Days poor mental health (0–30; 88=None, 77=DK, 99=Refused)
    "AVEDRNK3",   # Average drinks per drinking day (1–76; 88/77/99=special)
    "SLEPTIM1",   # Avg hours of sleep per 24h (1–24; 77/99=special)
]

# ── Demographic features (12) ──────────────────────────────────────────────
DEMO_FEATURES: List[str] = [
    "_AGEG5YR",   # 5-year age group (7–13 after cohort filter)
    "SEX1",       # Sex (1=Male, 2=Female)
    "_RACE",      # Race/ethnicity (1–8)
    "_EDUCAG",    # Education level (1–4)
    "_INCOMG1",   # Income group (1–7)
    "MARITAL",    # Marital status (1–6)
    "EMPLOY1",    # Employment status (1–8)
    "RENTHOM1",   # Home ownership (1=Own, 2=Rent, 3=Other)
    "_METSTAT",   # Metropolitan status (1=Metro, 2=Non-metro)
    "CTYCODE1",   # County FIPS (retain as categorical)
    "_STATE",     # State FIPS
    "NUMHHOL4",   # Number in household (1–76+)
]

# ── Health behaviors (18) ──────────────────────────────────────────────────
BEHAVIOR_FEATURES: List[str] = [
    "SMOKE100",   # Smoked ≥100 cigarettes lifetime (1=Yes, 2=No)
    "_SMOKER3",   # Current smoker status (1–4)
    "_RFDRHV7",   # Heavy drinking (1=No, 2=Yes)
    "_TOTINDA",   # Leisure physical activity (1=Yes, 2=No)
    "EXERANY2",   # Any exercise past 30 days (1=Yes, 2=No)
    "STRENGTH",   # Strength training past 30 days (days 0–7+)
    "FRUIT2",     # Fruit consumption per day
    "FVGREEN1",   # Green vegetable consumption
    "FVORANG2",   # Orange-colored vegetable consumption
    "FVBEAN1",    # Bean/legume consumption
    "FVBPOTA1",   # Baked potato consumption
    "_FRUTSU1",   # Total fruit servings per day
    "_VEGESU1",   # Total vegetable servings per day
    "SSBSUGR3",   # Sugar-sweetened beverage days
    "ASNOW",      # Asthma currently (1=Yes, 2=No)
    "DRINKANY5",  # Had ≥1 drink in past 30 days (1=Yes, 2=No)
    "ALCDAY4",    # Alcohol drinking days past 30 days
    "BINGE5",     # Binge drinking episodes past 30 days
]

# ── Comorbidities (25) ─────────────────────────────────────────────────────
COMORBIDITY_FEATURES: List[str] = [
    "CVDINFR4",   # MI/heart attack ever (1=Yes, 2=No)
    "CVDCRHD4",   # Angina/coronary heart disease (1=Yes, 2=No)
    "CVDSTRK3",   # Stroke ever (1=Yes, 2=No)
    "ASTHMA3",    # Asthma ever (1=Yes, 2=No)
    "CHCSCNC1",   # Skin cancer ever (1=Yes, 2=No)
    "CHCOCNC1",   # Other cancer ever (1=Yes, 2=No)
    "CHCCOPD3",   # COPD/emphysema/chronic bronchitis (1=Yes, 2=No)
    "ADDEPEV3",   # Depressive disorder ever (1=Yes, 2=No)
    "CHCKDNY2",   # Kidney disease ever (1=Yes, 2=No)
    "DIABETE4",   # Diabetes (1=Yes, 2=Pre, 3=No, 4=Gestational)
    "HAVARTH5",   # Arthritis ever (1=Yes, 2=No)
    "LMTJOIN3",   # Joint pain/stiffness limiting activities (1=Yes, 2=No)
    "CIMEMLOS",   # (excluded from predictors — this is the outcome)
    "CIMEMOL2",   # Memory difficulties affect social/work activities
    "HIVRISK5",   # HIV risk behaviors (1=Yes, 2=No)
    "_MICHD",     # MI or CHD (computed, 1=Yes, 2=No)
    "WTCHSALT",   # Watching sodium intake (1–4)
    "HADMAM",     # Mammogram in past 2 years (1=Yes, 2=No; women)
    "PCPSAAD3",   # PSA test past 2 years (1=Yes, 2=No; men)
    "HPVTEST",    # HPV test (1=Yes, 2=No)
    "HYPERTEN",   # Hypertension computed flag
    "TOLDHI3",    # Told high cholesterol (1=Yes, 2=No)
    "CHOLCHK3",   # Cholesterol checked in past 5 years (1=Yes, 2=No)
    "_BMI5CAT",   # BMI category (1–4)
    "_RFBMI5",    # BMI overweight/obese flag (1=No, 2=Yes)
]
# Remove outcome variable if accidentally listed
COMORBIDITY_FEATURES = [v for v in COMORBIDITY_FEATURES if v != "CIMEMLOS"]

# ── Functional status (10) ─────────────────────────────────────────────────
FUNCTIONAL_FEATURES: List[str] = [
    "QLACTLM2",   # Activity limitations (1=Yes, 2=No)
    "USEEQUIP",   # Uses special equipment (1=Yes, 2=No)
    "BLIND",      # Blind/serious difficulty seeing (1=Yes, 2=No)
    "DECIDE",     # Difficulty concentrating/deciding (1=Yes, 2=No)
    "DIFFWALK",   # Difficulty walking/climbing stairs (1=Yes, 2=No)
    "DIFFDRES",   # Difficulty dressing/bathing (1=Yes, 2=No)
    "DIFFALON",   # Difficulty doing errands alone (1=Yes, 2=No)
    "FALL1",      # Fallen in past year (1=Yes, 2=No)
    "FALLINJ4",   # Injury from fall (1=Yes, 2=No)
    "GENHLTH",    # General health (1=Excellent … 5=Poor)
]

# ── Healthcare access (8) ──────────────────────────────────────────────────
HEALTHCARE_FEATURES: List[str] = [
    "PERSDOC3",   # Has personal doctor (1=Yes, one; 2=Yes, more; 3=No)
    "MEDCOST1",   # Cost prevented seeing doctor (1=Yes, 2=No)
    "CHECKUP1",   # Time since last routine checkup (1–4, 8=Never)
    "HLTHPLN1",   # Health plan coverage (1=Yes, 2=No)
    "PRIMINS1",   # Primary insurance type (1–8)
    "TELEMDI1",   # Telehealth visit past year (1=Yes, 2=No)
    "FLUSHOT7",   # Flu shot in past year (1=Yes, 2=No)
    "PNEUMVAC4",  # Pneumococcal vaccine (1=Yes, 2=No)
]

# ── Geographic/survey design (7) ──────────────────────────────────────────
GEO_FEATURES: List[str] = [
    "_URBSTAT",   # Urban/rural status (1=Urban, 2=Rural)
    "_METSTAT",   # Metro status (duplicate kept for domain completeness)
    "_REGION",    # US Census region (1–4)  [computed if absent]
    "HHADULT",    # Number of adults in household
    "NUMADULT",   # Number of adults (some years use NUMADULT vs HHADULT)
    "CHILDREN",   # Number of children (0–87+)
    "_STSTR",     # Stratum variable (for survey design)
]

# All 117 candidate features (de-duplicated)
ALL_FEATURES: List[str] = list(
    dict.fromkeys(
        SI_FEATURES
        + DEMO_FEATURES
        + BEHAVIOR_FEATURES
        + COMORBIDITY_FEATURES
        + FUNCTIONAL_FEATURES
        + HEALTHCARE_FEATURES
        + GEO_FEATURES
    )
)

# Ensure outcome is NOT in predictors
ALL_FEATURES = [f for f in ALL_FEATURES if f not in ("CIMEMLOS", "scd")]

# ---------------------------------------------------------------------------
# Special-value mappings
# (BRFSS uses 77=Don't Know, 88=None, 99=Refused for many continuous vars)
# ---------------------------------------------------------------------------
NONE_AS_ZERO_VARS: Dict[str, float] = {
    "MENTHLTH": 88,   # 88 means "None" → recode to 0
    "PHYSHLTH": 88,
    "POORHLTH": 88,
    "ALCDAY4": 888,
    "SLEPTIM1": None,  # no special 'none' code; keep as-is
}

DK_REFUSED: Dict[str, list] = {
    "MENTHLTH": [77, 99],
    "PHYSHLTH": [77, 99],
    "SLEPTIM1": [77, 99],
    "AVEDRNK3": [77, 99],
    "ALCDAY4": [777, 999],
    "STRENGTH": [777, 999],
    "GENHLTH": [7, 9],
    "QLACTLM2": [7, 9],
    "ADDEPEV3": [7, 9],
    "CVDSTRK3": [7, 9],
    "HAVARTH5": [7, 9],
    "DIFFWALK": [7, 9],
    "DECIDE": [7, 9],
}


# ---------------------------------------------------------------------------
# Recode helpers
# ---------------------------------------------------------------------------

def _recode_special(series: pd.Series, varname: str) -> pd.Series:
    """Recode BRFSS special values to NaN for a single variable."""
    s = pd.to_numeric(series, errors="coerce").copy()

    # Recode "None" as zero
    none_val = NONE_AS_ZERO_VARS.get(varname)
    if none_val is not None:
        s = s.replace(none_val, 0.0)

    # Recode Don't Know / Refused to NaN
    dk_vals = DK_REFUSED.get(varname, [])
    if dk_vals:
        s = s.replace(dk_vals, np.nan)

    return s


def _binary_recode(series: pd.Series, yes_val: int = 1, no_val: int = 2) -> pd.Series:
    """
    Recode a Yes/No BRFSS variable to 1/0; all other values → NaN.
    Most BRFSS dichotomous vars use 1=Yes, 2=No, 7=DK, 9=Refused.
    """
    s = pd.to_numeric(series, errors="coerce")
    recoded = pd.Series(np.nan, index=s.index)
    recoded[s == yes_val] = 1.0
    recoded[s == no_val] = 0.0
    return recoded


# ---------------------------------------------------------------------------
# Per-domain feature engineering
# ---------------------------------------------------------------------------

def _engineer_si_features(df: pd.DataFrame, out: pd.DataFrame) -> None:
    """Engineer / recode social isolation domain features in-place into *out*."""
    for var in SI_FEATURES:
        if var not in df.columns:
            out[var] = np.nan
            continue

        if var in ("MENTHLTH",):
            # 0–30 days; 88=None → 0; 77/99 → NaN
            s = _recode_special(df[var], var)
            out[var] = s.clip(0, 30)

        elif var in ("AVEDRNK3",):
            # 1–76 drinks; 88/77/99 → NaN
            s = _recode_special(df[var], var)
            out[var] = s.clip(0, 76)

        elif var in ("SLEPTIM1",):
            # 1–24 hours; 77/99 → NaN
            s = _recode_special(df[var], var)
            out[var] = s.clip(0, 24)

        elif var in ("SOCLACTV",):
            # 1=Yes → 1; 2=No → 0
            out[var] = _binary_recode(df[var])

        else:
            # Ordinal Likert (LSATISFY, EMTSUPRT, SDHISOLT, SDHBOND): 1–5 / 1–4
            s = pd.to_numeric(df[var], errors="coerce")
            # 7/9 → NaN
            s = s.replace([7, 9], np.nan)
            out[var] = s


def _engineer_demo_features(df: pd.DataFrame, out: pd.DataFrame) -> None:
    for var in DEMO_FEATURES:
        if var not in df.columns:
            out[var] = np.nan
            continue
        s = pd.to_numeric(df[var], errors="coerce")
        # Recode generic refusals
        s = s.replace([7, 9, 77, 99], np.nan)
        out[var] = s


def _engineer_comorbidity_features(df: pd.DataFrame, out: pd.DataFrame) -> None:
    # Most comorbidity vars are Yes(1)/No(2); recode to 1/0
    binary_vars = {
        "CVDINFR4", "CVDCRHD4", "CVDSTRK3", "ASTHMA3", "CHCSCNC1",
        "CHCOCNC1", "CHCCOPD3", "ADDEPEV3", "CHCKDNY2", "HAVARTH5",
        "LMTJOIN3", "CIMEMOL2", "HIVRISK5", "_MICHD", "_RFBMI5",
        "HADMAM", "HPVTEST", "TOLDHI3", "CHOLCHK3",
    }
    for var in COMORBIDITY_FEATURES:
        if var not in df.columns:
            out[var] = np.nan
            continue
        if var in binary_vars:
            out[var] = _binary_recode(df[var])
        else:
            s = pd.to_numeric(df[var], errors="coerce")
            s = s.replace([7, 9, 77, 99], np.nan)
            out[var] = s


def _engineer_behavior_features(df: pd.DataFrame, out: pd.DataFrame) -> None:
    binary_vars = {"SMOKE100", "EXERANY2", "_TOTINDA", "DRINKANY5", "ASNOW"}
    for var in BEHAVIOR_FEATURES:
        if var not in df.columns:
            out[var] = np.nan
            continue
        if var in binary_vars:
            out[var] = _binary_recode(df[var])
        else:
            s = _recode_special(df[var], var)
            out[var] = s


def _engineer_functional_features(df: pd.DataFrame, out: pd.DataFrame) -> None:
    binary_vars = {
        "QLACTLM2", "USEEQUIP", "BLIND", "DECIDE",
        "DIFFWALK", "DIFFDRES", "DIFFALON", "FALL1", "FALLINJ4",
    }
    for var in FUNCTIONAL_FEATURES:
        if var not in df.columns:
            out[var] = np.nan
            continue
        if var in binary_vars:
            out[var] = _binary_recode(df[var])
        else:
            s = pd.to_numeric(df[var], errors="coerce")
            s = s.replace([7, 9], np.nan)
            out[var] = s


def _engineer_healthcare_features(df: pd.DataFrame, out: pd.DataFrame) -> None:
    binary_vars = {"MEDCOST1", "HLTHPLN1", "TELEMDI1", "FLUSHOT7", "PNEUMVAC4"}
    for var in HEALTHCARE_FEATURES:
        if var not in df.columns:
            out[var] = np.nan
            continue
        if var in binary_vars:
            out[var] = _binary_recode(df[var])
        else:
            s = pd.to_numeric(df[var], errors="coerce")
            s = s.replace([7, 9, 77, 99], np.nan)
            out[var] = s


def _engineer_geo_features(df: pd.DataFrame, out: pd.DataFrame) -> None:
    for var in GEO_FEATURES:
        if var == "_REGION" and var not in df.columns:
            # Derive US Census region from _STATE FIPS if missing
            if "_STATE" in df.columns:
                state = pd.to_numeric(df["_STATE"], errors="coerce")
                region = pd.Series(np.nan, index=df.index)
                # Northeast: CT(9),ME(23),MA(25),NH(33),RI(44),VT(50),NJ(34),NY(36),PA(42)
                ne = {9, 23, 25, 33, 44, 50, 34, 36, 42}
                # Midwest: IL(17),IN(18),MI(26),OH(39),WI(55),IA(19),KS(20),MN(27),MO(29),NE(31),ND(38),SD(46)
                mw = {17, 18, 26, 39, 55, 19, 20, 27, 29, 31, 38, 46}
                # South: all others in South (simplified)
                # West: AK(2),AZ(4),CA(6),CO(8),HI(15),ID(16),MT(30),NV(32),NM(35),OR(41),UT(49),WA(53),WY(56)
                w = {2, 4, 6, 8, 15, 16, 30, 32, 35, 41, 49, 53, 56}
                region[state.isin(ne)] = 1
                region[state.isin(mw)] = 2
                region[state.isin(w)] = 4
                # Remaining (South) = 3
                region[~state.isin(ne | mw | w) & state.notna()] = 3
                out[var] = region
            else:
                out[var] = np.nan
        elif var not in df.columns:
            out[var] = np.nan
        else:
            out[var] = pd.to_numeric(df[var], errors="coerce")


# ---------------------------------------------------------------------------
# Derived / composite features
# ---------------------------------------------------------------------------

def _add_composite_features(df: pd.DataFrame, out: pd.DataFrame) -> None:
    """Add composite / derived features not directly in BRFSS."""

    # Social Isolation Composite Score (mean of available SI items, ordinal-encoded)
    si_cols = [c for c in SI_FEATURES if c in out.columns and out[c].notna().any()]
    if si_cols:
        # Scale each to 0–1 before averaging
        normed = pd.DataFrame(index=out.index)
        for c in si_cols:
            col = out[c]
            rng = col.max() - col.min()
            normed[c] = (col - col.min()) / rng if rng > 0 else 0.0
        out["SI_COMPOSITE"] = normed.mean(axis=1)

    # Multimorbidity count (number of comorbidities present)
    cm_binary = [
        c for c in [
            "CVDINFR4", "CVDCRHD4", "CVDSTRK3", "ASTHMA3", "CHCSCNC1",
            "CHCOCNC1", "CHCCOPD3", "ADDEPEV3", "CHCKDNY2", "HAVARTH5",
            "DIABETE4",
        ]
        if c in out.columns
    ]
    if cm_binary:
        out["MULTIMORBIDITY_N"] = out[cm_binary].fillna(0).sum(axis=1)

    # Physical inactivity flag
    if "_TOTINDA" in out.columns:
        out["INACTIVE"] = (out["_TOTINDA"] == 0).astype(float)

    # Adequate sleep flag (7–9 hours)
    if "SLEPTIM1" in out.columns:
        out["ADEQUATE_SLEEP"] = (
            out["SLEPTIM1"].between(7, 9, inclusive="both")
        ).astype(float)

    # Alcohol misuse flag (>2 avg drinks/day)
    if "AVEDRNK3" in out.columns:
        out["ALCOHOL_MISUSE"] = (out["AVEDRNK3"] > 2).astype(float)


# ---------------------------------------------------------------------------
# Master feature builder
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    include_composites: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Engineer all features from a raw BRFSS cohort DataFrame.

    Parameters
    ----------
    df                 : Output of data_utils.build_cohort().
    include_composites : Whether to include derived composite features.

    Returns
    -------
    (X, feature_names)
      X            – pd.DataFrame of shape (n_samples, n_features) with floats.
      feature_names – ordered list of column names in X.
    """
    out = pd.DataFrame(index=df.index)

    _engineer_si_features(df, out)
    _engineer_demo_features(df, out)
    _engineer_behavior_features(df, out)
    _engineer_comorbidity_features(df, out)
    _engineer_functional_features(df, out)
    _engineer_healthcare_features(df, out)
    _engineer_geo_features(df, out)

    if include_composites:
        _add_composite_features(df, out)

    # Drop columns that are entirely missing (e.g. variables not in this year)
    n_before = out.shape[1]
    out = out.dropna(axis=1, how="all")
    n_after = out.shape[1]
    if n_before != n_after:
        logger.info(
            "Dropped %d all-NaN columns; %d features remain", n_before - n_after, n_after
        )

    # Remove the outcome from X (safety guard)
    out = out.drop(columns=["scd", "CIMEMLOS"], errors="ignore")

    feature_names: List[str] = list(out.columns)
    logger.info(
        "Feature matrix: %d samples × %d features", len(out), len(feature_names)
    )

    return out, feature_names


# ---------------------------------------------------------------------------
# LASSO pre-screen
# ---------------------------------------------------------------------------

def lasso_prescreen(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 40,
    cv: int = 5,
    random_state: int = 42,
    si_features_always_keep: Optional[List[str]] = None,
) -> List[str]:
    """
    Reduce 117 candidate features to a subset via LassoCV coefficient magnitude.

    Social-isolation features (SI_FEATURES) are always retained regardless of
    LASSO selection to preserve the primary exposure set.

    Parameters
    ----------
    X                        : Feature matrix (NaN-imputed before LASSO).
    y                        : Binary outcome (0/1).
    n_features               : Target number of features to retain.
    cv                       : Cross-validation folds for LassoCV.
    random_state             : Reproducibility seed.
    si_features_always_keep  : Additional features to force-include.

    Returns
    -------
    List of selected feature names.
    """
    from sklearn.impute import SimpleImputer

    always_keep: List[str] = list(SI_FEATURES)  # primary exposure always retained
    if si_features_always_keep:
        always_keep += [f for f in si_features_always_keep if f not in always_keep]
    always_keep = [f for f in always_keep if f in X.columns]

    logger.info(
        "LASSO pre-screen: %d candidates → target %d features",
        X.shape[1],
        n_features,
    )

    # Impute NaN with median before LASSO
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    lasso = LassoCV(cv=cv, random_state=random_state, max_iter=10000, n_jobs=-1)
    lasso.fit(X_scaled, y)

    coef = np.abs(lasso.coef_)
    feature_names = list(X.columns)

    # Rank by |coefficient|; force-include SI features
    rank_idx = np.argsort(coef)[::-1]  # descending
    selected: List[str] = []
    for i in rank_idx:
        if len(selected) >= n_features and feature_names[i] not in always_keep:
            continue
        if feature_names[i] not in selected:
            selected.append(feature_names[i])

    # Ensure always-keep are in the list
    for f in always_keep:
        if f not in selected:
            selected.append(f)

    nonzero = int((coef > 0).sum())
    logger.info(
        "LASSO: alpha=%.6f, non-zero coefs=%d, selected=%d features (incl. %d SI)",
        lasso.alpha_,
        nonzero,
        len(selected),
        len(always_keep),
    )
    return selected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build feature matrix from processed BRFSS cohort.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to processed cohort Parquet file (output of data_utils).",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Path to save feature matrix as Parquet.",
    )
    p.add_argument(
        "--lasso",
        action="store_true",
        help="Run LASSO pre-screen and save selected feature list.",
    )
    p.add_argument(
        "--n-features",
        type=int,
        default=40,
        help="Target number of features to retain in LASSO pre-screen.",
    )
    p.add_argument(
        "--no-composites",
        action="store_true",
        help="Disable derived composite features.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    )

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Loading cohort from %s", args.input)
    df = pd.read_parquet(args.input)
    logger.info("Loaded %d rows", len(df))

    X, feature_names = build_features(df, include_composites=not args.no_composites)

    if args.lasso:
        if "scd" not in df.columns:
            raise ValueError("Column 'scd' not found in cohort — needed for LASSO.")
        y = df["scd"]
        selected = lasso_prescreen(X, y, n_features=args.n_features)
        X = X[selected]
        feature_names = selected
        # Save selected features list
        sel_path = Path(args.output).with_suffix(".features.json")
        with open(sel_path, "w") as fh:
            json.dump({"selected_features": selected, "n": len(selected)}, fh, indent=2)
        logger.info("Saved selected features to %s", sel_path)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Attach outcome and weight for convenience
    for col in ("scd", "survey_weight", "survey_year"):
        if col in df.columns:
            X[col] = df[col].values

    X.to_parquet(out_path, index=False, compression="snappy")
    logger.info(
        "Saved feature matrix: %d rows × %d features → %s",
        len(X),
        len(feature_names),
        out_path,
    )


if __name__ == "__main__":
    main()
