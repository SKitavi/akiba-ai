"""Tests for structured XGBoost SHAP explanations."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.features.build_features import FEATURE_COLUMNS
from src.model.predict import predict_risk_score, score_applicant
from src.xai.shap_explainer import (
    ContributionDirection,
    explain_prediction,
)


@pytest.fixture()
def training_data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(42)
    features = pd.DataFrame({name: rng.normal(size=60) for name in FEATURE_COLUMNS})
    labels = np.array([0, 1] * 30)
    return features, labels


@pytest.fixture()
def fitted_model(
    training_data: tuple[pd.DataFrame, np.ndarray],
) -> xgb.XGBClassifier:
    features, labels = training_data
    return xgb.XGBClassifier(
        n_estimators=12,
        max_depth=2,
        learning_rate=0.1,
        random_state=42,
        n_jobs=1,
    ).fit(features, labels)


def _applicant(training_data: tuple[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
    return training_data[0].iloc[[0]].copy()


def _stub_shap_values(monkeypatch: pytest.MonkeyPatch, values: list[float]) -> None:
    class StubExplainer:
        def __init__(self, model: xgb.XGBClassifier, model_output: str) -> None:
            assert model_output == "raw"

        def __call__(self, features: pd.DataFrame) -> SimpleNamespace:
            return SimpleNamespace(
                values=np.asarray([values], dtype=float),
                base_values=np.asarray([0.25]),
            )

    monkeypatch.setattr("src.xai.shap_explainer.shap.TreeExplainer", StubExplainer)


def test_explain_prediction_returns_all_feature_contributions(
    fitted_model: xgb.XGBClassifier,
    training_data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    explanation = explain_prediction(fitted_model, _applicant(training_data))

    assert len(explanation.contributions) == len(FEATURE_COLUMNS)
    assert [
        factor.feature_name for factor in explanation.contributions
    ] == FEATURE_COLUMNS
    assert 0.0 <= explanation.risk_score <= 1.0
    assert explanation.output_space == "raw_margin_log_odds"


def test_identifier_column_is_ignored(
    fitted_model: xgb.XGBClassifier,
    training_data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    applicant = _applicant(training_data)
    expected = explain_prediction(fitted_model, applicant)
    applicant.insert(0, "applicant_id", "APP_0001")

    actual = explain_prediction(fitted_model, applicant)

    assert actual.risk_score == pytest.approx(expected.risk_score)
    assert [item.feature_name for item in actual.contributions] == FEATURE_COLUMNS


def test_missing_features_fail_clearly(
    fitted_model: xgb.XGBClassifier,
    training_data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    applicant = _applicant(training_data).drop(columns=[FEATURE_COLUMNS[0]])

    with pytest.raises(ValueError, match="missing columns"):
        explain_prediction(fitted_model, applicant)


def test_exactly_one_applicant_is_required(
    fitted_model: xgb.XGBClassifier,
    training_data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        explain_prediction(fitted_model, training_data[0].iloc[:2])


def test_direction_separation_ranking_and_zero_handling(
    monkeypatch: pytest.MonkeyPatch,
    fitted_model: xgb.XGBClassifier,
    training_data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    values = [0.0] * len(FEATURE_COLUMNS)
    values[0] = 0.5
    values[1] = -0.8
    values[2] = 0.8
    values[3] = -0.2
    _stub_shap_values(monkeypatch, values)

    explanation = explain_prediction(fitted_model, _applicant(training_data), top_n=2)

    assert [item.feature_name for item in explanation.increasing_risk_factors] == [
        FEATURE_COLUMNS[2],
        FEATURE_COLUMNS[0],
    ]
    assert [item.feature_name for item in explanation.reducing_risk_factors] == [
        FEATURE_COLUMNS[1],
        FEATURE_COLUMNS[3],
    ]
    assert explanation.contributions[4].direction is ContributionDirection.NEUTRAL


def test_tie_ordering_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    fitted_model: xgb.XGBClassifier,
    training_data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    values = [0.0] * len(FEATURE_COLUMNS)
    values[0] = 0.5
    values[1] = 0.5
    _stub_shap_values(monkeypatch, values)

    explanation = explain_prediction(fitted_model, _applicant(training_data), top_n=2)
    expected = sorted(FEATURE_COLUMNS[:2])

    assert [
        item.feature_name for item in explanation.increasing_risk_factors
    ] == expected


@pytest.mark.parametrize("top_n", [0, 1, 50])
def test_top_n_edge_cases(
    top_n: int,
    monkeypatch: pytest.MonkeyPatch,
    fitted_model: xgb.XGBClassifier,
    training_data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    values = [1.0 if index % 2 == 0 else -1.0 for index in range(len(FEATURE_COLUMNS))]
    _stub_shap_values(monkeypatch, values)

    explanation = explain_prediction(
        fitted_model, _applicant(training_data), top_n=top_n
    )

    expected_count = min(top_n, len(FEATURE_COLUMNS) // 2)
    assert len(explanation.increasing_risk_factors) == expected_count
    assert len(explanation.reducing_risk_factors) == expected_count


@pytest.mark.parametrize("top_n", [-1, 1.5, True])
def test_invalid_top_n_fails(
    top_n: object,
    fitted_model: xgb.XGBClassifier,
    training_data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    error = (
        TypeError
        if not isinstance(top_n, int) or isinstance(top_n, bool)
        else ValueError
    )
    with pytest.raises(error):
        explain_prediction(fitted_model, _applicant(training_data), top_n=top_n)  # type: ignore[arg-type]


def test_model_schema_drift_is_detected(
    training_data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    features, labels = training_data
    renamed = features.rename(columns={FEATURE_COLUMNS[0]: "unexpected_feature"})
    model = xgb.XGBClassifier(n_estimators=3, n_jobs=1).fit(renamed, labels)

    with pytest.raises(ValueError, match="schema"):
        explain_prediction(model, features.iloc[[0]])


def test_non_finite_values_follow_prediction_policy(
    fitted_model: xgb.XGBClassifier,
    training_data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    applicant = _applicant(training_data)
    applicant.loc[applicant.index[0], FEATURE_COLUMNS[0]] = np.inf
    applicant.loc[applicant.index[0], FEATURE_COLUMNS[1]] = np.nan

    explanation = explain_prediction(fitted_model, applicant)

    assert explanation.contributions[0].feature_value == 0.0
    assert explanation.contributions[1].feature_value == 0.0


def test_saved_model_explanation_matches_established_score_path(
    tmp_path: Path,
    fitted_model: xgb.XGBClassifier,
    training_data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    model_path = tmp_path / "model.json"
    fitted_model.save_model(model_path)
    loaded_model = xgb.XGBClassifier()
    loaded_model.load_model(model_path)
    applicant = _applicant(training_data)

    explanation = explain_prediction(loaded_model, applicant)

    assert explanation.risk_score == pytest.approx(
        score_applicant(model_path, applicant)
    )
    assert explanation.risk_score == pytest.approx(
        predict_risk_score(loaded_model, applicant)
    )
