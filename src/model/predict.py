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
from src.model.loader import load_model_bundle


def prepare_model_features(applicant_features: pd.DataFrame) -> pd.DataFrame:
    """Validate and order applicant features for model inference.

    Extra columns, including ``applicant_id``, are intentionally excluded so
    identifiers cannot leak into the model input. Non-finite values follow the
    established prediction-time policy and are replaced with zero.

    Args:
        applicant_features: One or more rows containing every canonical model
                            feature listed in ``FEATURE_COLUMNS``.

    Returns:
        A numeric DataFrame containing only the canonical model features in
        their training order.

    Raises:
        TypeError: If ``applicant_features`` is not a pandas DataFrame.
        ValueError: If columns are duplicated, required features are missing,
                    no rows are supplied, or a feature is not numeric.
    """
    if not isinstance(applicant_features, pd.DataFrame):
        raise TypeError("applicant_features must be a pandas DataFrame.")
    if applicant_features.empty:
        raise ValueError("applicant_features must contain at least one row.")

    duplicate_columns = applicant_features.columns[
        applicant_features.columns.duplicated()
    ].tolist()
    if duplicate_columns:
        raise ValueError(
            f"applicant_features contains duplicate columns: {duplicate_columns}"
        )

    missing = [
        column for column in FEATURE_COLUMNS if column not in applicant_features.columns
    ]
    if missing:
        raise ValueError(f"applicant_features is missing columns: {missing}")

    ordered = applicant_features[FEATURE_COLUMNS].copy()
    try:
        ordered = ordered.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("All model feature values must be numeric.") from exc

    return ordered.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def predict_risk_score(
    model: xgb.XGBClassifier,
    applicant_features: pd.DataFrame,
) -> float:
    """Return the mean positive-class model score for prepared applicant rows.

    The result is the raw XGBoost class probability. The current model has no
    separate probability-calibration stage, so callers should not describe it
    as a calibrated probability.
    """
    features = prepare_model_features(applicant_features)
    probabilities = model.predict_proba(features)[:, 1]
    return float(np.mean(probabilities))


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
        XGBoost positive-class risk score in ``[0.0, 1.0]``. The current
        pipeline does not separately calibrate this value.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist on disk.
        ValueError:        If required feature columns are missing.
    """
    model_bundle = load_model_bundle(model_path)
    return predict_risk_score(model_bundle.model, applicant_features)
