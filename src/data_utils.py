"""
data_utils.py
=============
BRFSS data download, XPT parsing, cohort construction, and survey-weight
extraction for the Social Isolation & Subjective Cognitive Decline pipeline.

Locked cohort decisions
-----------------------
  Age filter   : _AGEG5YR >= 7  (corresponds to age 50+, categories 7–13)
  SCD outcome  : CIMEMLOS == 1 → SCD=1; CIMEMLOS == 2 → SCD=0;
                 CIMEMLOS in {7, 9} or missing → excluded
  Survey weight: _LLCPWT  (landline + cell-phone combined weight)
  Final N      : 192,567  (dev 121,970 BRFSS-2023; holdout 70,597 BRFSS-2024)

CLI
---
  python -m src.data_utils --year 2023 --data-dir data/raw --out data/processed/brfss2023.parquet
  python -m src.data_utils --year 2024 --data-dir data/raw --out data/processed/brfss2024.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BRFSS_BASE_URL = "https://www.cdc.gov/brfss/annual_data/{year}/files/LLCP{year}XPT.zip"

# CDC publishes files under slightly different names — map year → known filename
BRFSS_XPT_FILENAMES: dict[int, str] = {
    2023: "LLCP2023.XPT",
    2024: "LLCP2024.XPT",
}

# Age-group variable: categories 7–13 map to age 50+ (50–54, 55–59, …, 80+)
AGE_VAR = "_AGEG5YR"
AGE_50PLUS_MIN = 7

# Subjective Cognitive Decline ascertainment
SCD_VAR = "CIMEMLOS"
SCD_YES = 1          # "Yes" – confusion/memory loss that is getting worse
SCD_NO = 2           # "No"
SCD_EXCLUDE = {7, 9}  # Don't know / Refused

# Survey weight
WEIGHT_VAR = "_LLCPWT"

# FIPS state variable (for geographic stratification)
STATE_VAR = "_STATE"

# Year variable added by this pipeline
SURVEY_YEAR_VAR = "survey_year"

# Minimum required columns to consider a row parseable
REQUIRED_COLS = {AGE_VAR, SCD_VAR, WEIGHT_VAR}

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _url_for_year(year: int) -> str:
    """Return the CDC BRFSS ZIP download URL for *year*."""
    return BRFSS_BASE_URL.format(year=year)


def _md5(filepath: Path) -> str:
    """Return MD5 hex-digest of *filepath*."""
    h = hashlib.md5()
    with open(filepath, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_brfss(
    year: int,
    data_dir: str | Path = "data/raw",
    force: bool = False,
) -> Path:
    """
    Download the BRFSS XPT file for *year* from CDC.

    Parameters
    ----------
    year     : Survey year (2023 or 2024).
    data_dir : Local directory to store the raw file.
    force    : Re-download even if the file already exists.

    Returns
    -------
    Path to the extracted .XPT file.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    xpt_name = BRFSS_XPT_FILENAMES.get(year, f"LLCP{year}.XPT")
    xpt_path = data_dir / xpt_name

    if xpt_path.exists() and not force:
        logger.info("XPT file already present: %s", xpt_path)
        return xpt_path

    zip_url = _url_for_year(year)
    zip_path = data_dir / f"LLCP{year}.zip"

    logger.info("Downloading BRFSS %d from %s …", year, zip_url)
    try:
        urllib.request.urlretrieve(zip_url, zip_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download BRFSS {year} from {zip_url}: {exc}\n"
            "Please download manually from https://www.cdc.gov/brfss/annual_data/ "
            f"and place the XPT file at {xpt_path}."
        ) from exc

    # Unzip
    import zipfile

    logger.info("Extracting %s …", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    zip_path.unlink(missing_ok=True)  # tidy up

    if not xpt_path.exists():
        # Some years use uppercase/lowercase variants — find what was extracted
        candidates = list(data_dir.glob("*.XPT")) + list(data_dir.glob("*.xpt"))
        if candidates:
            xpt_path = candidates[0]
            logger.warning("Expected %s but found %s", xpt_name, xpt_path.name)
        else:
            raise FileNotFoundError(
                f"XPT file not found in {data_dir} after extraction. "
                "Check the zip contents manually."
            )

    logger.info("Downloaded: %s (md5=%s)", xpt_path, _md5(xpt_path))
    return xpt_path


# ---------------------------------------------------------------------------
# XPT loading
# ---------------------------------------------------------------------------

def load_xpt(filepath: str | Path, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Load a BRFSS SAS transport (.XPT) file into a DataFrame.

    Parameters
    ----------
    filepath : Path to the .XPT file.
    columns  : Optional list of column names to retain (load all if None).
               Column names are upper-cased automatically to match BRFSS naming.

    Returns
    -------
    pd.DataFrame with original column names preserved (upper-case).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"XPT file not found: {filepath}")

    logger.info("Loading XPT: %s", filepath)
    df = pd.read_sas(filepath, format="xport", encoding="utf-8")

    # BRFSS uses upper-case column names; enforce consistency
    df.columns = [c.upper() for c in df.columns]

    n_rows, n_cols = df.shape
    logger.info("Loaded %d rows × %d columns from %s", n_rows, n_cols, filepath.name)

    if columns is not None:
        # Keep requested columns that actually exist; warn about missing ones
        cols_upper = [c.upper() for c in columns]
        missing = [c for c in cols_upper if c not in df.columns]
        if missing:
            logger.warning("Requested columns not found in XPT: %s", missing)
        present = [c for c in cols_upper if c in df.columns]
        df = df[present]
        logger.info("Retained %d of %d requested columns", len(present), len(cols_upper))

    return df


# ---------------------------------------------------------------------------
# Cohort construction
# ---------------------------------------------------------------------------

def ascertain_scd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply SCD ascertainment from CIMEMLOS.

    Adds column ``scd`` (int8: 1 = SCD, 0 = No SCD).
    Rows with CIMEMLOS in {7, 9} or NaN are dropped.

    Returns
    -------
    DataFrame with ``scd`` column and excluded rows removed.
    """
    if SCD_VAR not in df.columns:
        raise KeyError(
            f"SCD variable '{SCD_VAR}' not found. "
            "Ensure CIMEMLOS is present in the BRFSS file."
        )

    n_before = len(df)
    # Convert to numeric (XPT may load as float)
    cimemlos = pd.to_numeric(df[SCD_VAR], errors="coerce")

    # Exclude don't-know / refused / missing
    valid_mask = cimemlos.isin([SCD_YES, SCD_NO])
    n_excluded = n_before - valid_mask.sum()
    logger.info(
        "SCD ascertainment: %d rows excluded (CIMEMLOS=7/9/missing) of %d",
        n_excluded,
        n_before,
    )

    df = df.loc[valid_mask].copy()
    df["scd"] = (cimemlos[valid_mask] == SCD_YES).astype("int8")

    prevalence = df["scd"].mean()
    logger.info(
        "SCD prevalence: %.1f%% (%d / %d)",
        prevalence * 100,
        df["scd"].sum(),
        len(df),
    )
    return df


def filter_age_50plus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict cohort to adults aged 50+ using _AGEG5YR >= 7.

    _AGEG5YR coding (BRFSS):
      1 = 18–24, 2 = 25–29, 3 = 30–34, 4 = 35–39, 5 = 40–44, 6 = 45–49,
      7 = 50–54, 8 = 55–59, 9 = 60–64, 10 = 65–69, 11 = 70–74,
      12 = 75–79, 13 = 80+,  14 = Don't know/Refused/Missing

    Returns
    -------
    DataFrame filtered to _AGEG5YR in [7, 13].
    """
    if AGE_VAR not in df.columns:
        raise KeyError(
            f"Age variable '{AGE_VAR}' not found. "
            "Ensure _AGEG5YR is present in the BRFSS file."
        )

    n_before = len(df)
    age_num = pd.to_numeric(df[AGE_VAR], errors="coerce")
    mask = (age_num >= AGE_50PLUS_MIN) & (age_num <= 13)
    df = df.loc[mask].copy()

    logger.info(
        "Age filter (50+): retained %d of %d rows (%.1f%%)",
        len(df),
        n_before,
        100 * len(df) / max(n_before, 1),
    )
    return df


def extract_survey_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and validate survey weights from _LLCPWT.

    - Rows with missing or non-positive weights are assigned the median weight.
    - Column is renamed to ``survey_weight`` for downstream clarity.

    Returns
    -------
    DataFrame with ``survey_weight`` column added.
    """
    if WEIGHT_VAR not in df.columns:
        logger.warning(
            "'%s' not found; assigning uniform weight = 1.0", WEIGHT_VAR
        )
        df["survey_weight"] = 1.0
        return df

    w = pd.to_numeric(df[WEIGHT_VAR], errors="coerce")
    invalid = w.isna() | (w <= 0)
    n_invalid = invalid.sum()

    if n_invalid > 0:
        median_w = w[~invalid].median()
        logger.warning(
            "%d rows have invalid weights; imputing median %.4f", n_invalid, median_w
        )
        w = w.fillna(median_w)
        w[w <= 0] = median_w

    df["survey_weight"] = w.values
    logger.info(
        "Survey weights: min=%.2f, median=%.2f, max=%.2f",
        w.min(),
        w.median(),
        w.max(),
    )
    return df


def build_cohort(df: pd.DataFrame, year: Optional[int] = None) -> pd.DataFrame:
    """
    Apply the full cohort inclusion/exclusion pipeline in order:
      1. Age filter (50+)
      2. SCD ascertainment (CIMEMLOS)
      3. Survey weight extraction

    Parameters
    ----------
    df   : Raw BRFSS DataFrame (from load_xpt).
    year : Survey year label added as ``survey_year`` column (optional).

    Returns
    -------
    Analysis-ready DataFrame with ``scd`` and ``survey_weight`` columns.
    """
    # Validate required columns
    missing_req = REQUIRED_COLS - set(df.columns)
    if missing_req:
        raise KeyError(
            f"Required columns missing from BRFSS data: {missing_req}. "
            "Check that the full BRFSS XPT file was loaded."
        )

    df = filter_age_50plus(df)
    df = ascertain_scd(df)
    df = extract_survey_weights(df)

    if year is not None:
        df[SURVEY_YEAR_VAR] = year

    logger.info("Final cohort size: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Convenience loader (download + load + filter)
# ---------------------------------------------------------------------------

def load_brfss(
    year: int,
    data_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    force_download: bool = False,
    force_rebuild: bool = False,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    End-to-end loader: download (if needed) → parse XPT → build cohort → cache.

    The processed cohort is saved as a Parquet file for fast reloads.

    Parameters
    ----------
    year           : 2023 (development) or 2024 (holdout).
    data_dir       : Directory for raw XPT files.
    processed_dir  : Directory for processed Parquet cache.
    force_download : Re-download even if XPT exists.
    force_rebuild  : Re-process even if Parquet cache exists.
    columns        : Subset of columns to load (loads all if None).

    Returns
    -------
    Analysis-ready pd.DataFrame.
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_path = processed_dir / f"brfss{year}_cohort.parquet"

    if cache_path.exists() and not force_rebuild:
        logger.info("Loading cached cohort from %s", cache_path)
        df = pd.read_parquet(cache_path)
        logger.info("Cached cohort: %d rows", len(df))
        return df

    xpt_path = download_brfss(year, data_dir, force=force_download)
    df_raw = load_xpt(xpt_path, columns=columns)
    df = build_cohort(df_raw, year=year)

    df.to_parquet(cache_path, index=False, compression="snappy")
    logger.info("Cohort cached to %s", cache_path)
    return df


# ---------------------------------------------------------------------------
# Dev / holdout split
# ---------------------------------------------------------------------------

def get_dev_holdout(
    data_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    columns: Optional[list[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and return (dev_2023, holdout_2024) DataFrames.

    Expected sizes: dev ≈ 121,970; holdout ≈ 70,597; combined ≈ 192,567.

    Returns
    -------
    Tuple of (df_dev, df_holdout).
    """
    df_dev = load_brfss(
        2023, data_dir=data_dir, processed_dir=processed_dir, columns=columns
    )
    df_holdout = load_brfss(
        2024, data_dir=data_dir, processed_dir=processed_dir, columns=columns
    )

    logger.info(
        "Dev set (2023): %d  |  Holdout set (2024): %d  |  Combined: %d",
        len(df_dev),
        len(df_holdout),
        len(df_dev) + len(df_holdout),
    )
    return df_dev, df_holdout


# ---------------------------------------------------------------------------
# Descriptive summary
# ---------------------------------------------------------------------------

def describe_cohort(df: pd.DataFrame) -> dict:
    """
    Return a dictionary of summary statistics for a cohort DataFrame.

    Useful for validation and reporting.
    """
    summary: dict = {}
    summary["n"] = len(df)

    if "scd" in df.columns:
        summary["scd_n"] = int(df["scd"].sum())
        summary["scd_pct"] = round(float(df["scd"].mean()) * 100, 1)

    if SURVEY_YEAR_VAR in df.columns:
        summary["survey_year"] = sorted(df[SURVEY_YEAR_VAR].unique().tolist())

    if AGE_VAR in df.columns:
        vc = pd.to_numeric(df[AGE_VAR], errors="coerce").value_counts().sort_index()
        summary["age_group_counts"] = vc.to_dict()

    if "survey_weight" in df.columns:
        w = df["survey_weight"]
        summary["weight_min"] = round(float(w.min()), 2)
        summary["weight_median"] = round(float(w.median()), 2)
        summary["weight_max"] = round(float(w.max()), 2)

    if STATE_VAR in df.columns:
        summary["n_states"] = int(pd.to_numeric(df[STATE_VAR], errors="coerce").nunique())

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download, parse, and filter BRFSS data for the SCD pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--year",
        type=int,
        choices=[2023, 2024],
        required=True,
        help="BRFSS survey year to process.",
    )
    p.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory for raw XPT files.",
    )
    p.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory for processed Parquet cache.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional explicit output Parquet path (overrides processed-dir default).",
    )
    p.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download even if XPT file exists.",
    )
    p.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Re-process even if Parquet cache exists.",
    )
    p.add_argument(
        "--describe",
        action="store_true",
        help="Print cohort summary statistics and exit.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Set log level to DEBUG.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    df = load_brfss(
        year=args.year,
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        force_download=args.force_download,
        force_rebuild=args.force_rebuild,
    )

    if args.describe:
        import json

        summary = describe_cohort(df)
        print(json.dumps(summary, indent=2))
        return

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False, compression="snappy")
        logger.info("Saved cohort to %s", out_path)

    logger.info("Done. Final cohort: %d rows.", len(df))


if __name__ == "__main__":
    main()
