"""Tests for deterministic English and Kiswahili risk narratives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.features.build_features import FEATURE_COLUMNS
from src.xai.narratives import (
    FEATURE_LABELS,
    NarrativeLanguage,
    build_narrative,
    generate_risk_narrative,
    get_feature_label,
)
from src.xai.shap_explainer import (
    ContributionDirection,
    FeatureContribution,
    PredictionExplanation,
    explain_prediction,
)


def _factor(
    feature_name: str,
    shap_value: float,
    direction: ContributionDirection,
) -> FeatureContribution:
    return FeatureContribution(
        feature_name=feature_name,
        feature_value=1.5,
        shap_value=shap_value,
        direction=direction,
    )


def _explanation(
    increasing: tuple[FeatureContribution, ...] | None = None,
    reducing: tuple[FeatureContribution, ...] | None = None,
) -> PredictionExplanation:
    increasing = increasing or ()
    reducing = reducing or ()
    return PredictionExplanation(
        risk_score=0.4321,
        base_value=-0.5,
        output_space="raw_margin_log_odds",
        contributions=increasing + reducing,
        increasing_risk_factors=increasing,
        reducing_risk_factors=reducing,
    )


@pytest.fixture()
def directional_explanation() -> PredictionExplanation:
    return _explanation(
        increasing=(
            _factor(
                "low_balance_rate",
                0.7,
                ContributionDirection.INCREASES_RISK,
            ),
        ),
        reducing=(
            _factor(
                "inflow_regularity",
                -0.5,
                ContributionDirection.REDUCES_RISK,
            ),
        ),
    )


def test_english_narrative_is_structured_and_directional(
    directional_explanation: PredictionExplanation,
) -> None:
    narrative = generate_risk_narrative(directional_explanation, "en")

    assert narrative.language is NarrativeLanguage.ENGLISH
    assert "risk score of 0.432" in narrative.summary
    assert "higher estimated risk" in narrative.increasing_risk_factors[0].text
    assert "lower estimated risk" in narrative.reducing_risk_factors[0].text


def test_kiswahili_narrative_is_localized(
    directional_explanation: PredictionExplanation,
) -> None:
    narrative = generate_risk_narrative(directional_explanation, "sw")

    assert narrative.language is NarrativeLanguage.KISWAHILI
    assert "alama ya hatari ya 0.432" in narrative.summary
    assert "hatari kubwa" in narrative.increasing_risk_factors[0].text
    assert "hatari ndogo" in narrative.reducing_risk_factors[0].text
    assert "data sanisi" in narrative.disclaimer


def test_every_model_feature_has_both_human_readable_labels() -> None:
    assert set(FEATURE_LABELS) == set(FEATURE_COLUMNS)
    for feature_name in FEATURE_COLUMNS:
        assert get_feature_label(feature_name, "en") != feature_name
        assert get_feature_label(feature_name, "sw") != feature_name


def test_unknown_feature_uses_readable_fallback() -> None:
    factor = _factor(
        "new_behavior_signal",
        0.2,
        ContributionDirection.INCREASES_RISK,
    )

    narrative = generate_risk_narrative(_explanation(increasing=(factor,)), "en")

    rendered = narrative.increasing_risk_factors[0]
    assert rendered.feature_name == "new_behavior_signal"
    assert rendered.feature_label == "New behavior signal"
    assert "new_behavior_signal" not in rendered.text


@pytest.mark.parametrize("language", ["fr", "rw", "", "english"])
def test_unsupported_language_fails_clearly(language: str) -> None:
    with pytest.raises(ValueError, match="Unsupported language"):
        generate_risk_narrative(_explanation(), language)


def test_generation_is_deterministic(
    directional_explanation: PredictionExplanation,
) -> None:
    first = generate_risk_narrative(directional_explanation, "en")
    second = generate_risk_narrative(directional_explanation, "en")

    assert first == second


def test_empty_increasing_collection_is_preserved() -> None:
    reducing = (_factor("mean_balance", -0.3, ContributionDirection.REDUCES_RISK),)
    narrative = generate_risk_narrative(_explanation(reducing=reducing), "en")

    assert narrative.increasing_risk_factors == ()
    assert len(narrative.reducing_risk_factors) == 1


def test_empty_reducing_collection_is_preserved() -> None:
    increasing = (
        _factor(
            "negative_net_months",
            0.3,
            ContributionDirection.INCREASES_RISK,
        ),
    )
    narrative = generate_risk_narrative(_explanation(increasing=increasing), "sw")

    assert len(narrative.increasing_risk_factors) == 1
    assert narrative.reducing_risk_factors == ()


@pytest.mark.parametrize("language", ["en", "sw"])
def test_disclaimer_covers_model_behavior_synthetic_data_and_human_review(
    language: str,
) -> None:
    disclaimer = generate_risk_narrative(_explanation(), language).disclaimer.lower()

    expected_terms = (
        ("model behavior", "synthetic", "human review")
        if language == "en"
        else ("tabia ya modeli", "data sanisi", "mapitio ya binadamu")
    )
    assert all(term in disclaimer for term in expected_terms)


def test_legacy_build_narrative_no_longer_raises_not_implemented() -> None:
    result = build_narrative(
        {"low_balance_rate": 0.5, "inflow_regularity": -0.2},
        language="en",
    )

    assert "higher estimated risk" in result
    assert "lower estimated risk" in result
    assert "not causation" in result


@pytest.mark.parametrize("language", ["en", "sw"])
def test_trained_model_to_localized_narrative_integration(language: str) -> None:
    rng = np.random.default_rng(7)
    features = pd.DataFrame(
        {feature: rng.normal(size=40) for feature in FEATURE_COLUMNS}
    )
    labels = np.array([0, 1] * 20)
    model = xgb.XGBClassifier(
        n_estimators=8,
        max_depth=2,
        random_state=42,
        n_jobs=1,
    ).fit(features, labels)

    explanation = explain_prediction(model, features.iloc[[0]], top_n=3)
    narrative = generate_risk_narrative(explanation, language)

    assert narrative.risk_score == pytest.approx(explanation.risk_score)
    assert len(narrative.increasing_risk_factors) <= 3
    assert len(narrative.reducing_risk_factors) <= 3
    assert narrative.summary
    assert narrative.disclaimer
