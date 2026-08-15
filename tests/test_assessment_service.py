"""Tests for Streamlit-independent applicant assessment orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.application.assessment import (
    ApplicantScopeError,
    AssessmentExecutionError,
    AssessmentInputError,
    FeatureEngineeringError,
    NoTransactionsError,
    assess_applicant,
)
from src.domain.transactions import (
    NormalizedTransaction,
    TransactionProvider,
    TransactionType,
)
from src.features.build_features import FEATURE_COLUMNS
from src.model.loader import ModelBundle, load_model_bundle
from src.xai.narratives import NarrativeLanguage


@pytest.fixture()
def model_bundle(tmp_path: Path) -> ModelBundle:
    rng = np.random.default_rng(12)
    features = pd.DataFrame(
        {feature: rng.normal(size=50) for feature in FEATURE_COLUMNS}
    )
    labels = np.array([0, 1] * 25)
    model = xgb.XGBClassifier(
        n_estimators=8, max_depth=2, random_state=42, n_jobs=1
    ).fit(features, labels)
    model_path = tmp_path / "model.json"
    model.save_model(str(model_path))
    model_path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "model_version": "assessment_test_v1",
                "n_features": len(FEATURE_COLUMNS),
                "feature_columns": FEATURE_COLUMNS,
            }
        ),
        encoding="utf-8",
    )
    return load_model_bundle(model_path)


def _transactions(applicant_id: str = "APP_0001") -> tuple[NormalizedTransaction, ...]:
    return (
        NormalizedTransaction(
            applicant_id=applicant_id,
            timestamp=pd.Timestamp("2026-01-01 08:00:00").to_pydatetime(),
            provider=TransactionProvider.MPESA,
            tx_type=TransactionType.CASH_IN,
            amount=5000.0,
            post_balance=7000.0,
        ),
        NormalizedTransaction(
            applicant_id=applicant_id,
            timestamp=pd.Timestamp("2026-01-08 09:00:00").to_pydatetime(),
            provider=TransactionProvider.MPESA,
            tx_type=TransactionType.P2P_SEND,
            amount=1000.0,
            post_balance=6000.0,
        ),
        NormalizedTransaction(
            applicant_id=applicant_id,
            timestamp=pd.Timestamp("2026-02-01 08:00:00").to_pydatetime(),
            provider=TransactionProvider.MPESA,
            tx_type=TransactionType.P2P_RECEIVE,
            amount=3000.0,
            post_balance=9000.0,
        ),
    )


def test_valid_assessment_returns_complete_typed_result(
    model_bundle: ModelBundle,
) -> None:
    result = assess_applicant(_transactions(), model_bundle, top_n=3)

    assert result.applicant_id == "APP_0001"
    assert result.model_version == "assessment_test_v1"
    assert 0.0 <= result.risk_score <= 1.0
    assert len(result.features) == len(FEATURE_COLUMNS)
    assert result.explanation.risk_score == pytest.approx(result.risk_score)
    assert result.narrative.language is NarrativeLanguage.ENGLISH
    assert not hasattr(result, "decision")


def test_kiswahili_assessment(model_bundle: ModelBundle) -> None:
    result = assess_applicant(_transactions(), model_bundle, language="sw")

    assert result.narrative.language is NarrativeLanguage.KISWAHILI
    assert "alama ya hatari" in result.narrative.summary


def test_explicit_applicant_id_is_attached(model_bundle: ModelBundle) -> None:
    result = assess_applicant(
        _transactions("APP_SELECTED"),
        model_bundle,
        applicant_id="APP_SELECTED",
    )

    assert result.applicant_id == "APP_SELECTED"


def test_empty_transactions_fail(model_bundle: ModelBundle) -> None:
    with pytest.raises(NoTransactionsError, match="At least one"):
        assess_applicant((), model_bundle)


def test_invalid_applicant_fails(model_bundle: ModelBundle) -> None:
    with pytest.raises(ApplicantScopeError, match="requested applicant"):
        assess_applicant(_transactions(), model_bundle, applicant_id="APP_OTHER")


def test_mixed_applicants_fail(model_bundle: ModelBundle) -> None:
    mixed = _transactions() + _transactions("APP_0002")

    with pytest.raises(ApplicantScopeError, match="exactly one"):
        assess_applicant(mixed, model_bundle)


def test_missing_applicant_metadata_fails(model_bundle: ModelBundle) -> None:
    applicants = pd.DataFrame(
        [{"applicant_id": "APP_OTHER", "avg_monthly_income": 10000.0}]
    )

    with pytest.raises(AssessmentInputError, match="missing from applicants_df"):
        assess_applicant(_transactions(), model_bundle, applicants_df=applicants)


def test_unsupported_language_fails(model_bundle: ModelBundle) -> None:
    with pytest.raises(AssessmentInputError, match="Unsupported language"):
        assess_applicant(_transactions(), model_bundle, language="fr")


def test_unfitted_model_failure_is_wrapped(model_bundle: ModelBundle) -> None:
    invalid_bundle = ModelBundle(
        model=xgb.XGBClassifier(),
        model_path=model_bundle.model_path,
        model_version="invalid",
        metadata={},
        feature_names=tuple(FEATURE_COLUMNS),
        schema_verified=False,
    )

    with pytest.raises(AssessmentExecutionError, match="score and explain"):
        assess_applicant(_transactions(), invalid_bundle)


def test_feature_engineering_failure_is_wrapped(
    monkeypatch: pytest.MonkeyPatch, model_bundle: ModelBundle
) -> None:
    def fail_feature_building(*args: object, **kwargs: object) -> pd.DataFrame:
        raise ValueError("synthetic feature failure")

    monkeypatch.setattr(
        "src.application.assessment.build_feature_table", fail_feature_building
    )

    with pytest.raises(FeatureEngineeringError, match="Could not build features"):
        assess_applicant(_transactions(), model_bundle)
