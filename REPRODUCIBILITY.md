# Reproducibility Guide

**Repository:** Social Isolation and SCD — BRFSS 2023–2024 ML Study
**Manuscript:** *JMIR AI* (under review)

This document provides a complete step-by-step guide to reproduce all analyses,
tables, and figures reported in the manuscript, from raw BRFSS data download
through final model evaluation and figure generation.

---

## Prerequisites

### System requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| Python   | 3.9     | 3.11        |
| RAM      | 8 GB    | 16 GB       |
| Disk     | 5 GB    | 10 GB       |
| CPU      | 4 cores | 8+ cores    |
| GPU      | Not required | Not required |

### Install dependencies

```bash
git clone https://github.com/[USERNAME]/social-isolation-scd-brfss.git
cd social-isolation-scd-brfss
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Verify installation

```bash
python -c "import xgboost, shap, optuna, pandas, scipy; print('All packages OK')"
```

---

## Step 0: Directory Setup

After cloning, create the data and results directories if not already present:

```bash
mkdir -p data/raw data/processed results/models results/shap results/metrics
```

---

## Step 1: Download BRFSS Data

BRFSS data are publicly available from the CDC. Each year is distributed as a
**SAS XPT transport file** (`LLCP2023.XPT` and `LLCP2024.XPT`).

### Option A — Automated download script

```bash
bash data/download_brfss.sh
```

Expected download sizes:
- `LLCP2023.XPT` — ~450 MB
- `LLCP2024.XPT` — ~460 MB

### Option B — Manual download

1. Navigate to https://www.cdc.gov/brfss/annual_data/annual_2023.html
2. Under "Data", click **2023 BRFSS Data (XPT)** → save as `data/raw/LLCP2023.XPT`
3. Navigate to https://www.cdc.gov/brfss/annual_data/annual_2024.html
4. Under "Data", click **2024 BRFSS Data (XPT)** → save as `data/raw/LLCP2024.XPT`

### Codebook reference

The BRFSS codebooks document all variable names, codes, and response categories:
- 2023: https://www.cdc.gov/brfss/annual_data/2023/pdf/codebook23_llcp-v2-508.pdf
- 2024: https://www.cdc.gov/brfss/annual_data/2024/pdf/codebook24_llcp-508.pdf

### Verify download

```python
import pandas as pd
df23 = pd.read_sas("data/raw/LLCP2023.XPT", format="xport")
df24 = pd.read_sas("data/raw/LLCP2024.XPT", format="xport")
print(f"2023: {len(df23):,} rows × {df23.shape[1]} cols")
print(f"2024: {len(df24):,} rows × {df24.shape[1]} cols")
# Expected: 2023 ~433,000 rows; 2024 ~450,000 rows (before age/SCD filtering)
```

---

## Step 2: Data Preprocessing

**Notebook:** `notebooks/02_preprocessing.ipynb`
**Script:** `src/data_utils.py`

### Key preprocessing steps

1. **Age filter:** Restrict to respondents aged ≥ 50 years (`_AGE65YR` or calculated from `_AGEG5YR`)
2. **SCD ascertainment:** Outcome variable from `CIMEMLOS` (Thinking and Memory Module):
   - `CIMEMLOS = 1` (Yes) → SCD = 1
   - `CIMEMLOS = 2` (No) → SCD = 0
   - `CIMEMLOS = 7, 9, missing` → Excluded
3. **Survey weight variable:** `_LLCPWT` (final rake weight for national estimates)
4. **Stratum / PSU:** `_STSTR`, `_PSU` (retained for complex survey variance estimation)
5. **Temporal split:** BRFSS 2023 → development set; BRFSS 2024 → holdout set

### Expected output sizes after preprocessing

| Cohort | N (raw aged 50+) | N after SCD filter | SCD prevalence |
|--------|------------------|--------------------|----------------|
| BRFSS 2023 (development) | ~170,000 | **121,970** | 17.0% |
| BRFSS 2024 (holdout)     | ~180,000 | **70,597**  | 17.0% |
| **Combined**             |          | **192,567** | **17.0%** |

```bash
# Run preprocessing
python src/data_utils.py --year 2023 --out data/processed/brfss2023_clean.parquet
python src/data_utils.py --year 2024 --out data/processed/brfss2024_clean.parquet
```

---

## Step 3: Feature Engineering

**Notebook:** `notebooks/03_feature_engineering.ipynb`
**Script:** `src/features.py`

### Feature domains

| Domain | # Features | Key variables |
|--------|-----------|---------------|
| Social isolation | 8 | `LSATISFY`, `EMTSUPRT`, `SDHISOLT`, `SDHBOND`, social club attendance |
| Demographics | 12 | Age, sex, race/ethnicity, education, income, marital status |
| Health behaviors | 18 | Smoking, alcohol, physical activity, sleep duration |
| Comorbidities | 31 | Depression (`ADDEPEV3`), diabetes, CVD, hearing loss, vision loss |
| Functional status | 14 | ADL limitations, mobility, cognitive activity level |
| Healthcare access | 10 | Insurance, having a personal doctor, cost barriers |
| Geographic | 7 | Rural/urban classification, state-level variables |
| Survey design | 17 | `_STSTR`, `_PSU`, `_LLCPWT` (not used as model features, retained) |

### LASSO pre-screening

A LASSO logistic regression (CV-tuned λ) is used to reduce the 117-variable candidate
pool before XGBoost training. Variables with non-zero LASSO coefficients are retained.
This step is embedded in `notebooks/03_feature_engineering.ipynb`.

```bash
python src/features.py \
  --input data/processed/brfss2023_clean.parquet \
  --out data/processed/brfss2023_features.parquet \
  --lasso-screen
```

---

## Step 4: Model Training

**Notebook:** `notebooks/04_model_training.ipynb`
**Script:** `src/models.py`

### XGBoost hyperparameter tuning

Bayesian hyperparameter search with Optuna (100 trials, 5-fold stratified CV):

```python
# Key hyperparameter search space
param_space = {
    "n_estimators":       [100, 1000],
    "max_depth":          [3, 9],
    "learning_rate":      [0.01, 0.3],
    "subsample":          [0.6, 1.0],
    "colsample_bytree":   [0.6, 1.0],
    "min_child_weight":   [1, 10],
    "gamma":              [0, 5],
    "reg_alpha":          [0, 1],
    "reg_lambda":         [1, 10],
    "scale_pos_weight":   [1, 5],   # class imbalance correction
}
```

```bash
python src/models.py \
  --train data/processed/brfss2023_features.parquet \
  --holdout data/processed/brfss2024_features.parquet \
  --out results/models/ \
  --n-trials 100
```

### Platt calibration

After training, predicted probabilities are recalibrated using Platt scaling
(logistic regression of raw XGBoost scores vs. labels on a held-out calibration
slice of the development set).

### Baseline comparisons

The following baselines are trained for AUROC comparison (Table 3):

| Model | Implementation |
|-------|---------------|
| Logistic regression | `sklearn.linear_model.LogisticRegression` |
| Decision tree | `sklearn.tree.DecisionTreeClassifier` |
| Random forest | `sklearn.ensemble.RandomForestClassifier` |
| Gradient boosting (sklearn) | `sklearn.ensemble.GradientBoostingClassifier` |
| Naïve Bayes | `sklearn.naive_bayes.GaussianNB` |
| XGBoost (final) | `xgboost.XGBClassifier` |

### Expected outputs

```
results/models/
├── xgb_best_params.json        # Optuna best hyperparameters
├── xgb_model.ubj               # Saved XGBoost model (binary format)
├── xgb_calibrated.pkl          # Platt-calibrated model wrapper
├── xgb_dev_preds.parquet       # Development set predicted probabilities
├── xgb_holdout_preds.parquet   # Holdout set predicted probabilities
└── baseline_aurocs.csv         # AUROC table for all baselines
```

---

## Step 5: SHAP Analysis and MSII Construction

**Notebook:** `notebooks/05_shap_msii.ipynb`
**Script:** `src/shap_utils.py`

### SHAP computation

```python
import shap
explainer    = shap.TreeExplainer(xgb_model)
shap_values  = explainer(X_holdout)   # shape: (70597, n_features)
```

> **Runtime note:** SHAP computation on the full holdout set (70,597 rows) takes
> approximately 8–15 minutes on an 8-core CPU.

### MSII Construction

The Modifiable Social Isolation Index (MSII) is a SHAP-weighted composite of
the 8 social isolation features:

```python
# Social isolation feature names (BRFSS variable codes)
SI_FEATURES = [
    "LSATISFY",    # Satisfaction with life
    "EMTSUPRT",    # Emotional support received
    "SDHISOLT",    # Social isolation (direct SDOH question)
    "SDHBOND",     # Sense of belonging
    "SOCLACTV",    # Social/recreational activity participation
    "MENTHLTH",    # Mental health days (proxy)
    "AVEDRNK3",    # Drinking frequency (social context proxy)
    "SLEPTIM1",    # Sleep duration (social isolation proxy)
]

# MSII = mean |SHAP| across SI features, then standardized
shap_si      = shap_values[:, SI_FEATURES]
msii_raw     = shap_si.values.mean(axis=1)
msii_z       = (msii_raw - msii_raw.mean()) / msii_raw.std()
```

### MSII ablation

AUROC is computed with and without the MSII features to quantify the index's
marginal contribution. Expected: ΔAUROC = +0.0063.

### Expected outputs

```
results/shap/
├── shap_values_holdout.npy      # Raw SHAP matrix (70597 × n_features)
├── shap_summary_top20.csv       # Mean |SHAP| for top-20 features
├── msii_scores_holdout.parquet  # MSII scores (raw + standardized)
└── msii_ablation.csv            # AUROC with/without MSII features
```

---

## Step 6: Model Evaluation

**Notebook:** `notebooks/06_evaluation.ipynb`
**Script:** `src/evaluation.py`

### Primary metrics (reproduce Table 2)

```python
from src.evaluation import compute_metrics
metrics = compute_metrics(y_true, y_prob, threshold=0.1697)
# Expected output:
# AUROC   = 0.7678  (95% CI: 0.7595–0.7761)
# AUPRC   = 0.4339
# Brier   = 0.1193
# Sens    = 0.720
# Spec    = 0.680
# PPV     = 0.310
# NPV     = 0.920
# NNS     = 3.2
```

### Bootstrap confidence intervals

All 95% CIs are computed with 1,000 bootstrap replicates (stratified by outcome):

```python
from src.evaluation import bootstrap_ci
auroc_ci = bootstrap_ci(y_true, y_prob, metric="auroc", n_boot=1000, seed=42)
```

### Subgroup analysis (reproduce Table 4)

Subgroups evaluated on holdout set:

| Subgroup | Expected AUROC |
|----------|---------------|
| Age 50–64 | 0.8172 (95% CI 0.8091–0.8253) |
| Age 65–74 | ~0.77 |
| Age 75–79 | ~0.74 |
| Age ≥80   | 0.7031 (95% CI 0.6900–0.7161) |
| Male      | ~0.76 |
| Female    | ~0.77 |
| NH White  | ~0.77 |
| NH Black  | ~0.74 |

Cochran's Q heterogeneity test: Q = 202.6, df = 7, p < 0.001

### Survey-weighted AUROC

```python
from src.evaluation import survey_weighted_auroc
sw_auroc = survey_weighted_auroc(y_true, y_prob, weights=llcpwt)
# Expected: 0.7678 (ΔAUROC = 0.0000 vs. unweighted)
```

### Decision curve analysis

```python
from src.evaluation import decision_curve_analysis
dca = decision_curve_analysis(y_true, y_prob, threshold_range=(0.05, 0.50))
```

### Hosmer-Lemeshow calibration test

```python
from src.evaluation import hosmer_lemeshow
hl_stat, hl_p = hosmer_lemeshow(y_true, y_prob, n_bins=10)
```

---

## Step 7: Figure Generation

**Notebook:** `notebooks/07_figures.ipynb`
**Script:** `src/figures.py`

All manuscript figures are regenerated by running:

```bash
python src/figures.py --holdout-preds results/models/xgb_holdout_preds.parquet \
                      --shap results/shap/shap_values_holdout.npy \
                      --outdir figures/
```

| Figure | File | Description |
|--------|------|-------------|
| Figure 1 | `Figure1_flowchart.png` | CONSORT-style inclusion flowchart |
| Figure 2 | `Figure2_ROC_curves.png` | ROC curves (holdout + 8 subgroups) |
| Figure 3a | `Figure3_Calibration.png` | Calibration plot (Platt-scaled) |
| Figure 3b | `Figure3_SHAP_beeswarm.png` | SHAP beeswarm (top-20 features) |
| Figure 4 | `Figure4_DCA.png` | Decision curve analysis |

---

## Locked Analytic Dataset Checksum

To verify that the processed data matches the published analysis exactly,
the following SHA-256 checksums are provided for the processed parquet files:

```
# Run: sha256sum data/processed/brfss2023_features.parquet
brfss2023_features.parquet:  [CHECKSUM — to be added upon acceptance]

# Run: sha256sum data/processed/brfss2024_features.parquet
brfss2024_features.parquet:  [CHECKSUM — to be added upon acceptance]
```

> Checksums will be finalized and published with the accepted manuscript.

---

## Troubleshooting

| Issue | Likely cause | Fix |
|-------|-------------|-----|
| `MemoryError` during SHAP | Holdout set too large for RAM | Set `shap_sample_n = 10000` in config |
| `KeyError: 'CIMEMLOS'` | BRFSS year missing Cognitive Module | Check that full BRFSS (not landline-only) was downloaded |
| `XPT parse error` | Incomplete download | Re-download the XPT file; verify size ≥ 400 MB |
| AUROC ≠ 0.7678 | Different random seed or Optuna trial count | Set `seed=42`, `n_trials=100` in config |
| Survey weight mismatch | `_LLCPWT` column not found | Confirm `_LLCPWT` column present; some BRFSS files use `LLCPWT` without underscore |

---

## Runtime Estimates

| Step | Estimated time (8-core CPU) |
|------|-----------------------------|
| BRFSS download | 10–20 min (network dependent) |
| Preprocessing (both years) | 5–10 min |
| Feature engineering + LASSO | 3–5 min |
| XGBoost hyperparameter tuning (100 trials) | 20–40 min |
| SHAP computation (holdout, 70,597 rows) | 8–15 min |
| Evaluation + DCA + bootstrap CIs | 5–10 min |
| Figure generation | 2–3 min |
| **Total** | **~45–90 min** |

---

## Contact

For reproducibility issues, please open a GitHub Issue with:
1. Your Python and package versions (`pip freeze > requirements_your_env.txt`)
2. The error traceback
3. Which step/notebook failed

For manuscript-related questions, contact the corresponding author:
**[CORRESPONDING AUTHOR EMAIL]**
