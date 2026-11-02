"""
Social Isolation and SCD — BRFSS 2023-2024 ML Pipeline.

Public API
----------
>>> from src.data_utils  import load_and_preprocess
>>> from src.features    import build_features
>>> from src.models      import train_xgboost, PlattXGB
>>> from src.shap_utils  import compute_shap, build_msii, msii_ablation
>>> from src.evaluation  import compute_metrics, bootstrap_ci, survey_weighted_auroc
"""

__version__ = "1.0.0"
__author__  = "[AUTHOR NAME(S)]"
