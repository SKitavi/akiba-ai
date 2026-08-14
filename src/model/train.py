"""Model training entrypoint for the AkibaAI MVP.

Fits a baseline XGBoost binary-classification model for credit risk scoring.

Expected input ``features_df`` schema
--------------------------------------
Must contain all columns listed in ``src.features.build_features.FEATURE_COLUMNS``
plus a ``default_label`` column (0 = healthy, 1 = default).

Produced artifact
-----------------
``model_output_path`` receives the serialised XGBoost booster in JSON format.
A companion ``<model_output_path>.meta.json`` sidecar stores training metadata
(feature list, threshold, version).

Owner: Sharon (ML Engineer).
Sprint day due: Day 4 (Aug 13) - model training + eval milestone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from src.features.build_features import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

MODEL_VERSION = "xgb_v1"

#: XGBoost hyperparameters (conservative baseline for a 250-applicant dataset)
_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 1.0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 4,   # ~1/default_rate to handle 20% class imbalance
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_model(
    features_df: pd.DataFrame,
    model_output_path: Path,
) -> dict[str, Any]:
    """Train an MVP XGBoost model and persist serialized artifact(s).

    Performs 5-fold stratified cross-validation to report held-out AUC, then
    re-fits on the full dataset and saves the booster to ``model_output_path``.

    Args:
        features_df:       DataFrame with feature columns (see ``FEATURE_COLUMNS``)
                           and a ``default_label`` target column.
        model_output_path: ``Path`` where the XGBoost JSON artifact is written.
                           Parent directories are created automatically.

    Returns:
        Metrics dictionary with keys:
          - ``model_version``  : version tag string
          - ``n_samples``      : training set size
          - ``n_features``     : number of input features
          - ``cv_auc_mean``    : mean cross-validated ROC-AUC
          - ``cv_auc_std``     : std of cross-validated ROC-AUC
          - ``default_rate``   : fraction of positive labels in training data
          - ``feature_columns``: list of feature names used

    Raises:
        ValueError: If required columns are missing from ``features_df``.
    """
    # --- Validate inputs ---------------------------------------------------
    missing_features = [c for c in FEATURE_COLUMNS if c not in features_df.columns]
    if missing_features:
        raise ValueError(f"features_df is missing feature columns: {missing_features}")
    if "default_label" not in features_df.columns:
        raise ValueError("features_df must contain a 'default_label' target column.")

    X = features_df[FEATURE_COLUMNS].copy()
    y = features_df["default_label"].astype(int)

    # Replace any NaN / inf values with column medians
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    # --- Cross-validation --------------------------------------------------
    model = xgb.XGBClassifier(**_XGB_PARAMS)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring="roc_auc",
        return_train_score=False,
    )
    cv_auc_mean = float(np.mean(cv_results["test_score"]))
    cv_auc_std = float(np.std(cv_results["test_score"]))

    # --- Full-dataset refit ------------------------------------------------
    model.fit(X, y)

    # --- Persist artifact --------------------------------------------------
    model_output_path = Path(model_output_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_output_path))

    # Write sidecar metadata
    metadata: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "n_samples": int(len(y)),
        "n_features": len(FEATURE_COLUMNS),
        "cv_auc_mean": round(cv_auc_mean, 4),
        "cv_auc_std": round(cv_auc_std, 4),
        "default_rate": round(float(y.mean()), 4),
        "feature_columns": FEATURE_COLUMNS,
    }
    meta_path = model_output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata
