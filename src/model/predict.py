"""Prediction-time model scoring helpers.

Loads a trained XGBoost artifact and scores an applicant feature vector.

Owner: Sharon (ML Engineer).
Sprint day due: Day 4 (Aug 13) - model training + eval milestone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.features.build_features import FEATURE_COLUMNS


def score_applicant(
    model_path: Path,
    applicant_features: pd.DataFrame,
) -> float:
    """Return a single applicant risk score between 0 and 1.

    Args:
        model_path:          Path to the saved XGBoost JSON model artifact
                             produced by ``train_model``.
        applicant_features:  Single-row (or multi-row) DataFrame containing all
                             columns listed in ``FEATURE_COLUMNS``.  When a
                             multi-row DataFrame is supplied the mean predicted
                             probability is returned (useful for ensembling).

    Returns:
        Calibrated probability of default in ``[0.0, 1.0]``.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist on disk.
        ValueError:        If required feature columns are missing.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    missing = [c for c in FEATURE_COLUMNS if c not in applicant_features.columns]
    if missing:
        raise ValueError(f"applicant_features is missing columns: {missing}")

    X = applicant_features[FEATURE_COLUMNS].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    proba = model.predict_proba(X)[:, 1]
    return float(np.mean(proba))
