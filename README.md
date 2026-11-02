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
A novel **Modifiable Social Isolation Index (MSII)** is constructed directly from SHAP values
to quantify the population-attributable contribution of social isolation to SCD risk.

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
| MSII adjusted prevalence ratio | 1.42 (95% CI 1.38–1.46) |
| Social isolation PAF | **17.1%** |
| Survey-weighted AUROC attenuation | ΔAUROC = 0.0000 |
| Subgroup AUROC (age 50–64) | 0.8172 (95% CI 0.8091–0.8253) |
| Subgroup AUROC (age ≥80) | 0.7031 (95% CI 0.6900–0.7161) |
| Subgroup heterogeneity (Cochran's Q) | Q = 202.6, df = 7, p < 0.001 |

---

## Repository Structure

```
social-isolation-scd-brfss/
├── README.md                     # This file
├── REPRODUCIBILITY.md            # Step-by-step reproduction guide
├── CITATION.cff                  # Citation metadata
├── LICENSE                       # MIT License
├── requirements.txt              # Python dependencies (pinned)
│
├── data/
│   └── README.md                 # Instructions to download BRFSS 2023 & 2024
│
├── notebooks/
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
│   ├── data_utils.py             # BRFSS download, XPT parsing, merging
│   ├── features.py               # Feature engineering pipeline
│   ├── models.py                 # XGBoost + baseline model definitions
│   ├── shap_utils.py             # SHAP wrappers & MSII construction
│   ├── evaluation.py             # Metrics, CI bootstrap, DCA, calibration
│   └── figures.py                # Matplotlib/seaborn figure generation
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

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/[USERNAME]/social-isolation-scd-brfss.git
cd social-isolation-scd-brfss
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download BRFSS data

See [`data/README.md`](data/README.md) for download instructions. BRFSS data are
publicly available from the CDC at no cost. Raw XPT files are **not** included in
this repository due to size (each year ~500 MB).

```bash
# Quick download (requires wget or curl)
./data/download_brfss.sh
```

### 4. Run the full pipeline

```bash
# Option A: Notebooks (recommended for exploration)
jupyter lab

# Option B: Command-line end-to-end
python src/run_pipeline.py --config config.yaml
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full step-by-step guide
with expected runtimes and checkpoints.

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
full feature inventory and Table S1 (supplementary) for variable coding.

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
