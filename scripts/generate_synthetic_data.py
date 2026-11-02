"""
generate_synthetic_data.py
==========================
Generate a synthetic BRFSS-like dataset that mirrors the column names,
variable codings, and approximate marginal distributions used in the
Social Isolation & SCD pipeline.

Usage
-----
  # Default: 200 000 rows, saved to data/synthetic/brfss_synthetic.parquet
  python scripts/generate_synthetic_data.py

  # Custom size and output path
  python scripts/generate_synthetic_data.py --n 50000 --out data/synthetic/brfss_small.parquet

  # CSV output
  python scripts/generate_synthetic_data.py --n 10000 --out data/synthetic/sample.csv

  # Reproducible run
  python scripts/generate_synthetic_data.py --seed 42

Notes
-----
* All values are drawn from calibrated distributions (not real respondent data).
* The synthetic dataset is designed for pipeline smoke-testing and development;
  it cannot reproduce the published results.
* Variable codings follow BRFSS 2023/2024 codebooks so that all src/ modules
  accept the synthetic file without modification.
* SCD prevalence is set to approximately 17% to match the analytic sample.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Variable specification
# ---------------------------------------------------------------------------
# Each entry: (column_name, distribution_type, params, missing_rate)
# distribution types: "choice", "poisson", "normal_clamp", "uniform_int", "weight"

VARIABLE_SPEC: list[tuple] = [
    # ── Outcome ─────────────────────────────────────────────────────────────
    # CIMEMLOS: 1=SCD (yes), 2=no SCD, 7=don't know, 9=refused
    # Marginal: ~17% SCD, ~2% missing/refused
    ("CIMEMLOS",   "choice",       {"vals": [1, 2, 7, 9],
                                    "p":    [0.170, 0.811, 0.012, 0.007]}, 0.00),

    # ── Age group (50+ only: categories 7–13) ───────────────────────────────
    # _AGEG5YR: 7=50-54, 8=55-59, 9=60-64, 10=65-69, 11=70-74, 12=75-79, 13=80+
    ("_AGEG5YR",   "choice",       {"vals": [7, 8, 9, 10, 11, 12, 13],
                                    "p":    [0.18, 0.16, 0.17, 0.16, 0.13, 0.10, 0.10]}, 0.00),

    # ── Social isolation features (MSII) ─────────────────────────────────────
    # LSATISFY: life satisfaction 1=Very sat, 2=Sat, 3=Dis, 4=Very dis, 9=refused
    ("LSATISFY",   "choice",       {"vals": [1, 2, 3, 4, 9],
                                    "p":    [0.38, 0.47, 0.10, 0.03, 0.02]}, 0.05),
    # EMTSUPRT: emotional support 1=Always, 2=Usually, 3=Sometimes, 4=Rarely, 5=Never
    ("EMTSUPRT",   "choice",       {"vals": [1, 2, 3, 4, 5, 9],
                                    "p":    [0.42, 0.28, 0.15, 0.08, 0.05, 0.02]}, 0.05),
    # SDHISOLT: social isolation 1=Always, 2=Usually, 3=Sometimes, 4=Rarely, 5=Never
    ("SDHISOLT",   "choice",       {"vals": [1, 2, 3, 4, 5, 9],
                                    "p":    [0.05, 0.08, 0.20, 0.35, 0.30, 0.02]}, 0.08),
    # SDHBOND: social bonding 1=Strongly agree, 2=Agree, 3=Disagree, 4=Strongly disagree
    ("SDHBOND",    "choice",       {"vals": [1, 2, 3, 4, 7, 9],
                                    "p":    [0.35, 0.42, 0.14, 0.05, 0.02, 0.02]}, 0.08),
    # SOCLACTV: social activities 1=More than once/week … 5=Never
    ("SOCLACTV",   "choice",       {"vals": [1, 2, 3, 4, 5, 9],
                                    "p":    [0.22, 0.27, 0.22, 0.18, 0.09, 0.02]}, 0.06),
    # MENTHLTH: poor mental health days (0–30); 88=none, 77=don't know, 99=refused
    ("MENTHLTH",   "choice",       {"vals": list(range(0, 31)) + [88, 77, 99],
                                    "p":    ([0.40] + [0.015]*10 + [0.012]*10 +
                                             [0.008]*10 + [0.08, 0.01, 0.01])}, 0.05),
    # AVEDRNK3: avg drinks per drinking day (1–76, 99=refused)
    ("AVEDRNK3",   "choice",       {"vals": list(range(1, 8)) + [77, 99],
                                    "p":    [0.30, 0.25, 0.18, 0.10, 0.06, 0.04, 0.03,
                                             0.02, 0.02]}, 0.35),
    # SLEPTIM1: sleep hours (1–24, 77=don't know, 99=refused)
    ("SLEPTIM1",   "choice",       {"vals": list(range(4, 11)) + [77, 99],
                                    "p":    [0.03, 0.10, 0.22, 0.28, 0.18,
                                             0.10, 0.05, 0.02, 0.02]}, 0.04),

    # ── Demographics ─────────────────────────────────────────────────────────
    # SEX1: 1=Male, 2=Female
    ("SEX1",       "choice",       {"vals": [1, 2], "p": [0.46, 0.54]}, 0.01),
    # _RACE: race/ethnicity (1=White, 2=Black, 3=AI/AN, 4=Asian, 5=NHPI, 6=Other, 7=Multi, 8=Hispanic, 9=Unknown)
    ("_RACE",      "choice",       {"vals": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                    "p":    [0.68, 0.11, 0.01, 0.04, 0.003,
                                             0.02, 0.02, 0.09, 0.017]}, 0.02),
    # _EDUCAG: education 1=<HS, 2=HS grad, 3=some college, 4=college+
    ("_EDUCAG",    "choice",       {"vals": [1, 2, 3, 4, 9],
                                    "p":    [0.10, 0.29, 0.28, 0.31, 0.02]}, 0.02),
    # _INCOMG1: income 1=<$15k … 8=>$200k, 9=missing
    ("_INCOMG1",   "choice",       {"vals": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                    "p":    [0.06, 0.09, 0.11, 0.12, 0.14,
                                             0.15, 0.16, 0.13, 0.04]}, 0.10),
    # MARITAL: 1=Married, 2=Divorced, 3=Widowed, 4=Separated, 5=Never, 6=Couple, 9=refused
    ("MARITAL",    "choice",       {"vals": [1, 2, 3, 4, 5, 6, 9],
                                    "p":    [0.51, 0.14, 0.15, 0.03, 0.10, 0.05, 0.02]}, 0.02),
    # _URBSTAT: 1=Urban, 2=Rural
    ("_URBSTAT",   "choice",       {"vals": [1, 2], "p": [0.80, 0.20]}, 0.00),

    # ── Comorbidities ────────────────────────────────────────────────────────
    # ADDEPEV3: depression diagnosis 1=Yes, 2=No, 7=DK, 9=Refused
    ("ADDEPEV3",   "choice",       {"vals": [1, 2, 7, 9],
                                    "p":    [0.22, 0.75, 0.02, 0.01]}, 0.02),
    # DIABETE4: diabetes 1=Yes, 2=Pre-DM, 3=No, 4=Gestational, 7=DK, 9=Refused
    ("DIABETE4",   "choice",       {"vals": [1, 2, 3, 4, 7, 9],
                                    "p":    [0.15, 0.08, 0.73, 0.001, 0.02, 0.009]}, 0.02),
    # CVDCRHD4: coronary heart disease 1=Yes, 2=No, 7=DK, 9=Refused
    ("CVDCRHD4",   "choice",       {"vals": [1, 2, 7, 9],
                                    "p":    [0.08, 0.89, 0.02, 0.01]}, 0.02),
    # CVDSTRK3: stroke 1=Yes, 2=No, 7=DK, 9=Refused
    ("CVDSTRK3",   "choice",       {"vals": [1, 2, 7, 9],
                                    "p":    [0.05, 0.93, 0.01, 0.01]}, 0.02),
    # HAVARTH5: arthritis 1=Yes, 2=No, 7=DK, 9=Refused
    ("HAVARTH5",   "choice",       {"vals": [1, 2, 7, 9],
                                    "p":    [0.38, 0.60, 0.01, 0.01]}, 0.02),
    # DEAF: hearing loss 1=Yes, 2=No, 7=DK, 9=Refused
    ("DEAF",       "choice",       {"vals": [1, 2, 7, 9],
                                    "p":    [0.17, 0.81, 0.01, 0.01]}, 0.03),

    # ── Health behaviors ─────────────────────────────────────────────────────
    # EXERANY2: exercise 1=Yes, 2=No, 7=DK, 9=Refused
    ("EXERANY2",   "choice",       {"vals": [1, 2, 7, 9],
                                    "p":    [0.72, 0.26, 0.01, 0.01]}, 0.02),
    # _SMOKER3: smoking 1=Every day, 2=Some days, 3=Former, 4=Never
    ("_SMOKER3",   "choice",       {"vals": [1, 2, 3, 4, 9],
                                    "p":    [0.08, 0.04, 0.36, 0.50, 0.02]}, 0.02),
    # DRNKANY6: any alcohol 1=Yes, 2=No, 7=DK, 9=Refused
    ("DRNKANY6",   "choice",       {"vals": [1, 2, 7, 9],
                                    "p":    [0.45, 0.52, 0.02, 0.01]}, 0.03),

    # ── Functional status ────────────────────────────────────────────────────
    # POORHLTH: poor physical/mental health days (0-30, 88=none, 77=DK, 99=refused)
    ("POORHLTH",   "choice",       {"vals": list(range(0, 31)) + [88, 77, 99],
                                    "p":    ([0.35] + [0.013]*10 + [0.010]*10 +
                                             [0.006]*10 + [0.09, 0.01, 0.01])}, 0.05),
    # PHYSHLTH: poor physical health days (same coding)
    ("PHYSHLTH",   "choice",       {"vals": list(range(0, 31)) + [88, 77, 99],
                                    "p":    ([0.40] + [0.012]*10 + [0.010]*10 +
                                             [0.006]*10 + [0.08, 0.01, 0.01])}, 0.05),
    # QLACTLM2: activity limitation 1=Yes, 2=No, 7=DK, 9=Refused
    ("QLACTLM2",   "choice",       {"vals": [1, 2, 7, 9],
                                    "p":    [0.28, 0.70, 0.01, 0.01]}, 0.02),
    # DIFFWALK: difficulty walking 1=Yes, 2=No, 7=DK, 9=Refused
    ("DIFFWALK",   "choice",       {"vals": [1, 2, 7, 9],
                                    "p":    [0.22, 0.76, 0.01, 0.01]}, 0.02),

    # ── Healthcare access ────────────────────────────────────────────────────
    # HLTHPLN2: health coverage 1=Yes, 2=No, 7=DK, 9=Refused
    ("HLTHPLN2",   "choice",       {"vals": [1, 2, 7, 9],
                                    "p":    [0.93, 0.06, 0.005, 0.005]}, 0.01),
    # PERSDOC3: personal doctor 1=Yes/only one, 2=Yes/more, 3=No, 7=DK, 9=Refused
    ("PERSDOC3",   "choice",       {"vals": [1, 2, 3, 7, 9],
                                    "p":    [0.68, 0.14, 0.16, 0.01, 0.01]}, 0.02),
    # MEDCOST1: couldn't see doctor due to cost 1=Yes, 2=No, 7=DK, 9=Refused
    ("MEDCOST1",   "choice",       {"vals": [1, 2, 7, 9],
                                    "p":    [0.08, 0.90, 0.01, 0.01]}, 0.02),
    # CHECKUP1: last routine checkup 1=<1yr, 2=1-<2yr, 3=2-<5yr, 4=5+yr, 8=Never, 9=Refused
    ("CHECKUP1",   "choice",       {"vals": [1, 2, 3, 4, 8, 9],
                                    "p":    [0.72, 0.13, 0.08, 0.04, 0.02, 0.01]}, 0.02),

    # ── Geographic ───────────────────────────────────────────────────────────
    # _STATE: 50 states + DC + territories (1–56, not all used)
    ("_STATE",     "choice",       {"vals": [1, 2, 4, 5, 6, 8, 9, 10, 11, 12,
                                             13, 15, 16, 17, 18, 19, 20, 21, 22,
                                             23, 24, 25, 26, 27, 28, 29, 30, 31,
                                             32, 33, 34, 35, 36, 37, 38, 39, 40,
                                             41, 42, 44, 45, 46, 47, 48, 49, 50,
                                             51, 53, 54, 55, 56],
                                    "p":    None}, 0.00),  # None = uniform

    # ── Survey weight ────────────────────────────────────────────────────────
    # _LLCPWT: design weight — log-normal to mimic BRFSS weight distribution
    ("_LLCPWT",    "weight",       {"mu": 3.5, "sigma": 1.2}, 0.00),

    # ── Year indicator (for dev/holdout split) ───────────────────────────────
    # Not a BRFSS variable; added by the pipeline to track the survey wave
    ("YEAR",       "choice",       {"vals": [2023, 2024],
                                    "p":    [0.633, 0.367]}, 0.00),  # ~121970/192567
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_synthetic_brfss(
    n: int = 200_000,
    seed: int = 0,
    scd_prevalence: float = 0.17,
) -> pd.DataFrame:
    """
    Return a synthetic BRFSS-like DataFrame with n rows.

    Parameters
    ----------
    n : int
        Number of rows (respondents) to generate.
    seed : int
        NumPy random seed for reproducibility.
    scd_prevalence : float
        Target prevalence of SCD (CIMEMLOS == 1). The raw marginals in
        VARIABLE_SPEC are adjusted post-hoc to match this target.

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(seed)
    logger.info("Generating %d synthetic BRFSS rows (seed=%d) …", n, seed)

    data: dict[str, np.ndarray] = {}

    for col, dist, params, missing_rate in VARIABLE_SPEC:
        if dist == "choice":
            vals = params["vals"]
            probs = params.get("p")
            if probs is not None:
                probs = np.array(probs, dtype=float)
                probs /= probs.sum()   # normalise
            arr = rng.choice(vals, size=n, p=probs)

        elif dist == "weight":
            arr = np.exp(rng.normal(params["mu"], params["sigma"], size=n))

        else:
            raise ValueError(f"Unknown distribution type: {dist}")

        # Inject missingness as NaN (float cast needed for int cols)
        if missing_rate > 0:
            mask = rng.random(n) < missing_rate
            arr = arr.astype(float)
            arr[mask] = np.nan

        data[col] = arr

    df = pd.DataFrame(data)

    # ── Post-hoc SCD prevalence correction ──────────────────────────────────
    # Ensure CIMEMLOS=1 (SCD) prevalence is close to target.
    current_scd = (df["CIMEMLOS"] == 1).mean()
    if abs(current_scd - scd_prevalence) > 0.02:
        logger.info(
            "Adjusting SCD prevalence from %.3f to %.3f …", current_scd, scd_prevalence
        )
        non_scd_idx = df.index[df["CIMEMLOS"] == 2].tolist()
        scd_idx     = df.index[df["CIMEMLOS"] == 1].tolist()
        target_scd  = int(n * scd_prevalence)
        if len(scd_idx) < target_scd:
            # Flip some non-SCD to SCD
            to_flip = rng.choice(non_scd_idx, size=target_scd - len(scd_idx), replace=False)
            df.loc[to_flip, "CIMEMLOS"] = 1
        else:
            # Flip some SCD back to non-SCD
            to_flip = rng.choice(scd_idx, size=len(scd_idx) - target_scd, replace=False)
            df.loc[to_flip, "CIMEMLOS"] = 2

    final_prev = (df["CIMEMLOS"] == 1).mean()
    logger.info("SCD prevalence in synthetic data: %.3f", final_prev)
    logger.info("Year breakdown: %s", df["YEAR"].value_counts().to_dict())

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a synthetic BRFSS-like dataset for pipeline testing."
    )
    p.add_argument("--n",    type=int,   default=200_000,
                   help="Number of rows to generate (default: 200 000)")
    p.add_argument("--seed", type=int,   default=0,
                   help="Random seed (default: 0)")
    p.add_argument("--out",  type=str,
                   default="data/synthetic/brfss_synthetic.parquet",
                   help="Output path (.parquet or .csv)")
    p.add_argument("--prevalence", type=float, default=0.17,
                   help="Target SCD prevalence (default: 0.17)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    out  = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_brfss(
        n=args.n,
        seed=args.seed,
        scd_prevalence=args.prevalence,
    )

    if out.suffix == ".csv":
        df.to_csv(out, index=False)
    else:
        df.to_parquet(out, index=False)

    logger.info("Saved %d rows × %d columns → %s", len(df), df.shape[1], out)
    logger.info(
        "Column list: %s", ", ".join(df.columns.tolist())
    )


if __name__ == "__main__":
    main()
