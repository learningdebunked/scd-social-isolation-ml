# Social Isolation and Subjective Cognitive Decline in 192,567 U.S. Adults Aged 50+
## An Interpretable Machine Learning Screening Study Using BRFSS 2023–2024

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![BRFSS Data](https://img.shields.io/badge/data-CDC%20BRFSS%202023%2F2024-green.svg)](https://www.cdc.gov/brfss/)
[![Journal: JMIR AI](https://img.shields.io/badge/journal-JMIR%20AI-orange.svg)](https://ai.jmir.org/)

---

## Overview

This repository contains all analysis code for the manuscript:

> **Social Isolation and Subjective Cognitive Decline in 192,567 U.S. Adults Aged 50+:
> An Interpretable Machine Learning Screening Study Using BRFSS 2023–2024**
>
> *[Author names — to be added upon acceptance]*
> *JMIR AI* (under review)

The study applies **XGBoost with SHAP-based interpretability** to the nationally representative
Behavioral Risk Factor Surveillance System (BRFSS), developing and prospectively validating
a machine learning screening model for subjective cognitive decline (SCD) in adults aged 50+.
A novel **Multi-dimensional Social Isolation Index (MSII)** is constructed from BRFSS social
isolation variables to quantify the incremental contribution of social isolation to SCD risk.

---

## Key Results

| Metric | Value |
|--------|-------|
| Total analytic sample | 192,567 (BRFSS 2023 + 2024) |
| Development cohort (BRFSS 2023) | 121,970 |
| Holdout cohort (BRFSS 2024) | 70,597 |
| SCD prevalence | 17.0% (32,655/192,567) |
| XGBoost holdout AUROC | **0.7678** (95% CI 0.7595–0.7761) |
| AUPRC | 0.4339 |
| Brier score | 0.1193 |
| Youden threshold | 0.1697 |
| Sensitivity / Specificity | 0.72 / 0.68 |
| PPV / NPV | 0.31 / 0.92 |
| Number needed to screen (NNS) | **3.2** |
| MSII ablation ΔAUROC | +0.0063 |
| MSII NRI_total | 0.218 |
| Survey-weighted AUROC attenuation | ΔAUROC = 0.0000 |
| Subgroup AUROC (age 50–64) | 0.8166 (95% CI 0.8099–0.8230) |
| Subgroup AUROC (age 65–79) | 0.7482 (95% CI 0.7404–0.7557) |
| Subgroup AUROC (age ≥80) | 0.6734 (95% CI 0.6593–0.6863) |
| Subgroup heterogeneity (Cochran's Q) | Q = 202.6, df = 7, p < 0.001 |
| Calibration | slope = 1.030; intercept = −0.072 |

---

## Repository Structure

```
scd-social-isolation-ml/
├── README.md                     # This file
├── REPRODUCIBILITY.md            # Step-by-step reproduction guide
├── CITATION.cff                  # Citation metadata
├── LICENSE                       # MIT License
├── requirements.txt              # Python dependencies (pinned)
│
├── data/
│   └── README.md                 # Instructions to download BRFSS 2023 & 2024
│
├── scripts/
│   └── generate_synthetic_data.py  # Synthetic BRFSS-like data generator
│
├── notebooks/
│   ├── 01_full_pipeline.ipynb    # End-to-end walkthrough (all src/ modules)
│   ├── 01_data_download.ipynb    # BRFSS XPT download & initial inspection
│   ├── 02_preprocessing.ipynb    # Data cleaning, recoding, survey weighting
│   ├── 03_feature_engineering.ipynb  # Feature construction & selection
│   ├── 04_model_training.ipynb   # XGBoost training & hyperparameter tuning
│   ├── 05_shap_msii.ipynb        # SHAP computation & MSII construction
│   ├── 06_evaluation.ipynb       # AUROC, calibration, DCA, subgroup analysis
│   └── 07_figures.ipynb          # All manuscript figures (Figures 1–5)
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py             # BRFSS download, XPT parsing, cohort building
│   ├── features.py               # Feature engineering pipeline
│   ├── models.py                 # XGBoost + baseline model definitions
│   ├── shap_utils.py             # SHAP wrappers & MSII construction
│   ├── evaluation.py             # Metrics, CI bootstrap, DCA, calibration
│   └── figures.py                # Matplotlib/seaborn figure generation
│
├── manuscript/
│   ├── loneliness_scd_manuscript.tex   # LaTeX source
│   ├── loneliness_scd_manuscript.pdf   # Compiled manuscript
│   ├── loneliness_scd_manuscript.docx  # Word version
│   ├── references.bib                  # BibTeX references
│   ├── venue.yaml                      # JMIR AI submission guidelines
│   ├── external_values.yaml            # Locked result manifest (Stage 6)
│   └── stage6_claim_guard.py           # Fact-lock consistency test (39/39 PASS)
│
├── results/
│   └── README.md                 # Description of saved results artifacts
│
└── figures/
    ├── Figure2_ROC_curves.png    # ROC curves (holdout + subgroups)
    ├── Figure3_Calibration.png   # Calibration plot (Platt-scaled)
    ├── Figure3_SHAP_beeswarm.png # SHAP beeswarm (top-20 features)
    └── Figure4_DCA.png           # Decision curve analysis
```

---

## How to Run

There are two ways to run this pipeline:

| Mode | When to use | Data required | Time |
|------|------------|---------------|------|
| **Synthetic data** | Development, testing, CI, demo | None — generated locally | ~5 min |
| **Real BRFSS data** | Full study reproduction | CDC BRFSS 2023 + 2024 XPT files (~1 GB) | ~45–90 min |

Both modes use the exact same `src/` pipeline code. The only difference is the
input parquet file passed to each module.

---

### Prerequisites (both modes)

#### 1. Clone the repository

```bash
git clone https://github.com/learningdebunked/scd-social-isolation-ml.git
cd scd-social-isolation-ml
```

#### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Python version:** 3.9 or higher required. Tested on 3.11.4.
> **RAM:** 8 GB minimum (16 GB recommended for the full BRFSS dataset).

---

### Option A — Synthetic Data (Quickstart, no downloads)

Use this path to explore the pipeline, test changes, or run in a CI environment
without downloading the 1 GB CDC files.

#### Step 1 — Generate a synthetic BRFSS dataset

```bash
# Default: 200,000 rows, ~17% SCD prevalence, saved to data/synthetic/
python scripts/generate_synthetic_data.py

# Smaller dataset for fast testing (recommended first run)
python scripts/generate_synthetic_data.py --n 20000 --out data/synthetic/brfss_small.parquet

# Custom seed for reproducibility
python scripts/generate_synthetic_data.py --n 200000 --seed 42

# CSV output instead of parquet
python scripts/generate_synthetic_data.py --n 50000 --out data/synthetic/brfss_50k.csv
```

The generator produces 36 BRFSS-coded columns (see table below) with realistic
marginal distributions and ~17% SCD prevalence. The `YEAR` column (2023/2024)
is used for the temporal dev/holdout split.

#### Step 2 — Run the full pipeline on synthetic data

```bash
# Build features
python -m src.features \
    --input  data/synthetic/brfss_synthetic.parquet \
    --output data/processed/features_synthetic.parquet

# Train XGBoost (uses YEAR column to split dev/holdout automatically)
python -m src.models \
    --input  data/processed/features_synthetic.parquet \
    --output results/model_synthetic/

# Compute SHAP values and MSII
python -m src.shap_utils \
    --model  results/model_synthetic/xgb_pipeline.pkl \
    --input  data/processed/features_synthetic.parquet \
    --output results/shap_synthetic/

# Evaluate on holdout set
python -m src.evaluation \
    --model  results/model_synthetic/xgb_pipeline.pkl \
    --input  data/processed/features_synthetic.parquet \
    --output results/evaluation_synthetic/
```

> **Note:** Synthetic data does not reproduce the published metrics. AUROC on
> synthetic data will typically be lower (~0.65–0.70) because the generator
> uses marginal distributions without real inter-variable correlations.

#### Step 3 — Interactive walkthrough (notebook)

```bash
jupyter lab notebooks/01_full_pipeline.ipynb
```

Set `DATA_PATH = "data/synthetic/brfss_synthetic.parquet"` in the first cell to
point the notebook at the synthetic file instead of the real BRFSS data.

---

### Option B — Real BRFSS Data (Full Reproduction)

Use this path to reproduce the published results exactly.

#### Step 1 — Download BRFSS data from CDC

BRFSS XPT files are publicly available, free, and require no registration.

```bash
# Manual download (recommended — verify checksums)
# BRFSS 2023:
wget -P data/raw/ \
  "https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip"

# BRFSS 2024:
wget -P data/raw/ \
  "https://www.cdc.gov/brfss/annual_data/2024/files/LLCP2024XPT.zip"

# Extract
unzip data/raw/LLCP2023XPT.zip -d data/raw/
unzip data/raw/LLCP2024XPT.zip -d data/raw/
```

Alternatively, use the built-in downloader:

```bash
python -m src.data_utils --year 2023 --data-dir data/raw \
    --out data/processed/brfss2023.parquet

python -m src.data_utils --year 2024 --data-dir data/raw \
    --out data/processed/brfss2024.parquet
```

Expected file sizes after extraction:

| File | Size |
|------|------|
| `LLCP2023.XPT` | ~487 MB |
| `LLCP2024.XPT` | ~501 MB |

#### Step 2 — Parse, filter, and merge cohorts

```bash
# Parse each XPT, apply age filter (50+), ascertain SCD outcome, extract weights
python -m src.data_utils \
    --year 2023 \
    --data-dir data/raw \
    --out data/processed/brfss2023.parquet

python -m src.data_utils \
    --year 2024 \
    --data-dir data/raw \
    --out data/processed/brfss2024.parquet
```

Expected output:

| File | Rows | SCD prevalence |
|------|------|----------------|
| `brfss2023.parquet` | 121,970 | 17.2% |
| `brfss2024.parquet` | 70,597 | 16.5% |

#### Step 3 — Build features

```bash
# Run feature engineering on 2023 (development) cohort
python -m src.features \
    --input  data/processed/brfss2023.parquet \
    --output data/processed/features_2023.parquet

# Run feature engineering on 2024 (holdout) cohort
python -m src.features \
    --input  data/processed/brfss2024.parquet \
    --output data/processed/features_2024.parquet
```

This step constructs 117 candidate features including the 5 MSII dimensions
and runs LASSO pre-screening to select the final feature set.

#### Step 4 — Train XGBoost and calibrate

```bash
python -m src.models \
    --train  data/processed/features_2023.parquet \
    --test   data/processed/features_2024.parquet \
    --output results/model/
```

This runs 5-fold stratified cross-validation with Optuna Bayesian hyperparameter
tuning, fits the final model, applies Platt scaling, and saves:

```
results/model/
├── xgb_pipeline.pkl        # Trained + calibrated XGBoost pipeline
├── model_metadata.json     # Hyperparameters, CV scores, training timestamp
└── cv_results.csv          # Per-fold AUROC scores
```

Expected CV AUROC (development): ~0.77. Expected holdout AUROC: **0.7678**.

> Runtime: ~20–40 minutes on a standard laptop (CPU only, no GPU required).

#### Step 5 — Compute SHAP values and MSII

```bash
python -m src.shap_utils \
    --model  results/model/xgb_pipeline.pkl \
    --input  data/processed/features_2024.parquet \
    --output results/shap/
```

Output:
```
results/shap/
├── shap_values.npy          # SHAP matrix (n_holdout × n_features)
├── shap_feature_names.json  # Feature name mapping
└── msii_scores.parquet      # Per-respondent MSII composite scores
```

#### Step 6 — Evaluate on holdout set

```bash
python -m src.evaluation \
    --model  results/model/xgb_pipeline.pkl \
    --input  data/processed/features_2024.parquet \
    --output results/evaluation/ \
    --shap   results/shap/shap_values.npy
```

This computes all reported metrics: AUROC with bootstrap CI, AUPRC, Brier score,
Youden threshold, calibration slope/intercept, decision curve analysis, and
subgroup AUROCs by age, sex, race/ethnicity, MSII level, and depression status.

Expected key outputs:

```
results/evaluation/
├── metrics_summary.json      # All primary metrics
├── subgroup_auroc.csv        # Subgroup breakdown
├── calibration_metrics.json  # Slope, intercept, H-L test
├── dca_results.csv           # Decision curve net-benefit values
└── figures/                  # Manuscript figures (Figures 1–5)
```

#### Step 7 — Reproduce all manuscript figures

```bash
python -m src.figures \
    --eval-dir  results/evaluation/ \
    --shap-dir  results/shap/ \
    --output    figures/
```

#### Step 8 — Run the Stage 6 fact-lock consistency check

```bash
python manuscript/stage6_claim_guard.py
# Expected: RESULT: ALL CHECKS PASS  (0 failures, 2 warning(s))
```

---

### Synthetic Data Variable Reference

The generator produces the following 36 columns, all following BRFSS codebook conventions:

| Column | Description | Values |
|--------|-------------|--------|
| `CIMEMLOS` | **Outcome**: SCD (confused/memory loss) | 1=Yes, 2=No, 7=DK, 9=Refused |
| `_AGEG5YR` | Age group (50+ only) | 7=50–54 … 13=80+ |
| `LSATISFY` | Life satisfaction | 1=Very sat … 4=Very dis |
| `EMTSUPRT` | Emotional support frequency | 1=Always … 5=Never |
| `SDHISOLT` | Social isolation frequency | 1=Always … 5=Never |
| `SDHBOND` | Social bonding | 1=Strongly agree … 4=Strongly disagree |
| `SOCLACTV` | Social activity frequency | 1=>1×/week … 5=Never |
| `MENTHLTH` | Poor mental health days | 0–30, 88=None |
| `AVEDRNK3` | Avg drinks per drinking day | 1–76 |
| `SLEPTIM1` | Sleep hours | 1–24 |
| `SEX1` | Biological sex | 1=Male, 2=Female |
| `_RACE` | Race/ethnicity | 1=White … 8=Hispanic |
| `_EDUCAG` | Education level | 1=<HS … 4=College+ |
| `_INCOMG1` | Household income group | 1=<$15k … 8=>$200k |
| `MARITAL` | Marital status | 1=Married … 5=Never |
| `_URBSTAT` | Urban/rural status | 1=Urban, 2=Rural |
| `ADDEPEV3` | Depression diagnosis | 1=Yes, 2=No |
| `DIABETE4` | Diabetes diagnosis | 1=Yes, 2=Pre-DM, 3=No |
| `CVDCRHD4` | Coronary heart disease | 1=Yes, 2=No |
| `CVDSTRK3` | Stroke | 1=Yes, 2=No |
| `HAVARTH5` | Arthritis | 1=Yes, 2=No |
| `DEAF` | Hearing loss | 1=Yes, 2=No |
| `EXERANY2` | Any exercise | 1=Yes, 2=No |
| `_SMOKER3` | Smoking status | 1=Every day … 4=Never |
| `DRNKANY6` | Any alcohol use | 1=Yes, 2=No |
| `POORHLTH` | Poor health days | 0–30, 88=None |
| `PHYSHLTH` | Poor physical health days | 0–30, 88=None |
| `QLACTLM2` | Activity limitation | 1=Yes, 2=No |
| `DIFFWALK` | Difficulty walking | 1=Yes, 2=No |
| `HLTHPLN2` | Health coverage | 1=Yes, 2=No |
| `PERSDOC3` | Personal doctor | 1=Yes/one … 3=No |
| `MEDCOST1` | Skipped care due to cost | 1=Yes, 2=No |
| `CHECKUP1` | Last routine checkup | 1=<1yr … 8=Never |
| `_STATE` | U.S. state FIPS code | 1–56 |
| `_LLCPWT` | Survey design weight | Positive float (log-normal) |
| `YEAR` | Survey wave | 2023 (dev), 2024 (holdout) |

---

## Data

**Source:** U.S. Centers for Disease Control and Prevention (CDC)

- **BRFSS 2023:** https://www.cdc.gov/brfss/annual_data/annual_2023.html
- **BRFSS 2024:** https://www.cdc.gov/brfss/annual_data/annual_2024.html

Data are publicly available, de-identified, and free to download. No registration
or data use agreement is required. See [`data/README.md`](data/README.md) for
variable-level documentation.

**Variables used:** A total of 117 candidate features were extracted, spanning
social isolation indicators, demographic characteristics, health behaviors,
comorbidities, and functional status. See Table 1 of the manuscript for the
full feature inventory.

---

## Methods Summary

| Step | Method | Implementation |
|------|--------|----------------|
| Preprocessing | Missing data imputation, BRFSS recode | `src/data_utils.py` |
| Feature engineering | 117 candidate features; LASSO pre-screen | `src/features.py` |
| Model | XGBoost (gradient boosted trees) | `xgboost==2.0.3` |
| Hyperparameter tuning | 5-fold stratified CV, Optuna Bayesian search | `optuna==3.5.0` |
| Interpretability | SHAP TreeExplainer | `shap==0.44.1` |
| MSII | SHAP-weighted social isolation composite index | `src/shap_utils.py` |
| Calibration | Platt scaling + Hosmer-Lemeshow test | `src/evaluation.py` |
| Decision curve | Net benefit across threshold range | `src/evaluation.py` |
| Survey weighting | `_LLCPWT` rake weights (CDC BRFSS design) | `src/evaluation.py` |
| Subgroup analysis | Age, sex, race/ethnicity, education, rural/urban | `notebooks/06_evaluation.ipynb` |
| Validation | Prospective temporal holdout (BRFSS 2024) | `notebooks/04_model_training.ipynb` |

---

## Computational Environment

All analyses were run on a standard laptop (no GPU required):

- **OS:** Ubuntu 22.04 LTS (also tested: macOS 13, Windows 11 via WSL2)
- **Python:** 3.11.4
- **RAM:** 16 GB (8 GB minimum for full pipeline)
- **Estimated total runtime:** ~45–90 minutes end-to-end (CPU only)

See `requirements.txt` for pinned package versions.

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{[AUTHOR]_2024_scd_brfss,
  author    = {[Author names]},
  title     = {Social Isolation and Subjective Cognitive Decline in 192,567
               U.S. Adults Aged 50+: An Interpretable Machine Learning
               Screening Study Using BRFSS 2023--2024},
  journal   = {JMIR AI},
  year      = {[YEAR]},
  doi       = {[DOI -- to be assigned upon acceptance]},
  issn      = {2817-1705}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

---

## License

This code is released under the [MIT License](LICENSE).

The underlying BRFSS data are in the public domain and distributed by the U.S.
Centers for Disease Control and Prevention. Use of BRFSS data is subject to CDC
data use guidelines.

---

## Contact

**Corresponding author:** [CORRESPONDING AUTHOR NAME]
**Email:** [CORRESPONDING AUTHOR EMAIL]
**Institution:** [INSTITUTION]

For questions about the analysis code, please open a GitHub Issue.
For questions about the manuscript, please contact the corresponding author directly.

---

## Acknowledgments

The Behavioral Risk Factor Surveillance System is conducted by the U.S. Centers for
Disease Control and Prevention (CDC). We thank the CDC and all state health
departments for making these data publicly available.
