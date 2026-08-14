"""Structured SHAP explanations for AkibaAI XGBoost risk scores.

SHAP values describe how model inputs move the XGBoost output away from its
baseline. They explain model behaviour and must not be interpreted as evidence
that a feature caused a real-world lending outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.features.build_features import FEATURE_COLUMNS
from src.model.predict import predict_risk_score, prepare_model_features


class ContributionDirection(str, Enum):
    """Direction in which a feature moves the model's raw risk output."""

    INCREASES_RISK = "increases_risk"
    REDUCES_RISK = "reduces_risk"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class FeatureContribution:
    """One model feature's value and local SHAP contribution."""

    feature_name: str
    feature_value: float
    shap_value: float
    direction: ContributionDirection

    @property
    def absolute_importance(self) -> float:
        """Return contribution magnitude without changing its direction."""
        return abs(self.shap_value)


@dataclass(frozen=True)
class PredictionExplanation:
    """Structured local explanation for one applicant model score.

    ``base_value`` and all ``shap_value`` fields use XGBoost's raw margin
    (log-odds) space. ``risk_score`` is retained separately as the model's
    positive-class probability-like score.
    """

    risk_score: float
    base_value: float
    output_space: str
    contributions: tuple[FeatureContribution, ...]
    increasing_risk_factors: tuple[FeatureContribution, ...]
    reducing_risk_factors: tuple[FeatureContribution, ...]


def _validate_model_schema(model: xgb.XGBClassifier) -> None:
    """Detect feature-order drift when the fitted booster stores feature names."""
    try:
        model_feature_names = model.get_booster().feature_names
    except (AttributeError, xgb.core.XGBoostError) as exc:
        raise ValueError("model must be a fitted XGBoost classifier.") from exc

    if model_feature_names is not None and list(model_feature_names) != FEATURE_COLUMNS:
        raise ValueError(
            "Model feature schema does not match the canonical FEATURE_COLUMNS order."
        )


def _direction(shap_value: float) -> ContributionDirection:
    if shap_value > 0.0:
        return ContributionDirection.INCREASES_RISK
    if shap_value < 0.0:
        return ContributionDirection.REDUCES_RISK
    return ContributionDirection.NEUTRAL


def _rank_factors(
    contributions: tuple[FeatureContribution, ...],
    direction: ContributionDirection,
    top_n: int,
) -> tuple[FeatureContribution, ...]:
    """Rank one contribution direction by magnitude with stable name ties."""
    matching = (factor for factor in contributions if factor.direction is direction)
    ranked = sorted(
        matching,
        key=lambda factor: (-factor.absolute_importance, factor.feature_name),
    )
    return tuple(ranked[:top_n])


def explain_prediction(
    model: xgb.XGBClassifier,
    features: pd.DataFrame,
    top_n: int = 5,
) -> PredictionExplanation:
    """Explain one applicant's XGBoost risk score with Tree SHAP.

    Args:
        model: A fitted ``XGBClassifier`` using the canonical 32-feature schema.
        features: Exactly one applicant row. Extra non-model columns are ignored.
        top_n: Maximum number of increasing and reducing factors to return.

    Returns:
        A typed explanation containing the model risk score, SHAP base value,
        all feature contributions in canonical order, and ranked directional
        factors. Zero contributions remain in ``contributions`` but are not
        included in either directional ranking.

    Raises:
        TypeError: If ``top_n`` is not an integer.
        ValueError: If input rows, features, model schema, or ``top_n`` are invalid.
        RuntimeError: If SHAP returns a shape incompatible with the pinned binary
                      XGBoost model contract.

    Notes:
        ``TreeExplainer`` is configured for the raw XGBoost output. Consequently,
        SHAP values are additive log-odds contributions, not percentage-point
        changes in predicted probability.
    """
    if isinstance(top_n, bool) or not isinstance(top_n, int):
        raise TypeError("top_n must be an integer.")
    if top_n < 0:
        raise ValueError("top_n must be greater than or equal to zero.")
    prepared = prepare_model_features(features)
    if len(prepared.index) != 1:
        raise ValueError("features must contain exactly one applicant row.")
    _validate_model_schema(model)

    explainer = shap.TreeExplainer(model, model_output="raw")
    shap_result = explainer(prepared)
    values = np.asarray(shap_result.values)
    base_values = np.asarray(shap_result.base_values).reshape(-1)

    expected_shape = (1, len(FEATURE_COLUMNS))
    if values.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected SHAP values shape {values.shape}; expected {expected_shape} "
            "for a binary XGBoost classifier."
        )
    if base_values.size != 1:
        raise RuntimeError(
            f"Unexpected SHAP base value shape {base_values.shape}; expected one value."
        )

    feature_row = prepared.iloc[0]
    contributions = tuple(
        FeatureContribution(
            feature_name=feature_name,
            feature_value=float(feature_row[feature_name]),
            shap_value=float(values[0, index]),
            direction=_direction(float(values[0, index])),
        )
        for index, feature_name in enumerate(FEATURE_COLUMNS)
    )

    return PredictionExplanation(
        risk_score=predict_risk_score(model, prepared),
        base_value=float(base_values[0]),
        output_space="raw_margin_log_odds",
        contributions=contributions,
        increasing_risk_factors=_rank_factors(
            contributions, ContributionDirection.INCREASES_RISK, top_n
        ),
        reducing_risk_factors=_rank_factors(
            contributions, ContributionDirection.REDUCES_RISK, top_n
        ),
    )


def compute_shap_values(
    model: xgb.XGBClassifier, features_df: pd.DataFrame
) -> np.ndarray:
    """Return the SHAP contribution matrix for compatibility with early callers.

    New application code should use :func:`explain_prediction`, which preserves
    the base value, risk score, contribution directions, and feature values.
    """
    prepared = prepare_model_features(features_df)
    _validate_model_schema(model)
    values = shap.TreeExplainer(model, model_output="raw")(prepared).values
    return np.asarray(values)
