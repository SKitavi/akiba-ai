"""Tests for atomic assessment and separate human-decision persistence."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pytest

from src.application.assessment import AssessmentResult
from src.features.build_features import FEATURE_COLUMNS
from src.storage.assessment_store import (
    AssessmentAuditContext,
    AssessmentPersistenceError,
    HumanDecision,
    persist_assessment,
    record_human_decision,
)
from src.storage.db import get_connection, initialize_schema, resolve_db_path
from src.xai.narratives import (
    NarrativeFactor,
    NarrativeLanguage,
    RiskNarrative,
)
from src.xai.shap_explainer import (
    ContributionDirection,
    FeatureContribution,
    PredictionExplanation,
)


_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "src" / "storage" / "schema.sql"


@pytest.fixture()
def connection(tmp_path: Path) -> sqlite3.Connection:
    database = get_connection(tmp_path / "assessment.db")
    initialize_schema(database, _SCHEMA_PATH)
    yield database
    database.close()


def _assessment() -> AssessmentResult:
    increasing = FeatureContribution(
        feature_name="low_balance_rate",
        feature_value=2.0,
        shap_value=0.4,
        direction=ContributionDirection.INCREASES_RISK,
    )
    reducing = FeatureContribution(
        feature_name="inflow_regularity",
        feature_value=0.9,
        shap_value=-0.3,
        direction=ContributionDirection.REDUCES_RISK,
    )
    explanation = PredictionExplanation(
        risk_score=0.42,
        base_value=-0.2,
        output_space="raw_margin_log_odds",
        contributions=(increasing, reducing),
        increasing_risk_factors=(increasing,),
        reducing_risk_factors=(reducing,),
    )
    narrative = RiskNarrative(
        language=NarrativeLanguage.ENGLISH,
        risk_score=0.42,
        summary="The model produced a risk score of 0.420.",
        increasing_risk_factors=(
            NarrativeFactor(
                feature_name=increasing.feature_name,
                feature_label="Frequency of low wallet balances",
                feature_value=increasing.feature_value,
                shap_value=increasing.shap_value,
                direction=increasing.direction,
                text="This factor moved the model toward higher estimated risk.",
            ),
        ),
        reducing_risk_factors=(
            NarrativeFactor(
                feature_name=reducing.feature_name,
                feature_label="Regularity of incoming funds",
                feature_value=reducing.feature_value,
                shap_value=reducing.shap_value,
                direction=reducing.direction,
                text="This factor moved the model toward lower estimated risk.",
            ),
        ),
        disclaimer="Model behavior, not causation; synthetic data; human review.",
    )
    return AssessmentResult(
        applicant_id="APP_0001",
        model_version="xgb_test_v1",
        risk_score=0.42,
        features={
            feature: float(index) for index, feature in enumerate(FEATURE_COLUMNS)
        },
        explanation=explanation,
        narrative=narrative,
    )


def test_schema_contains_explanations_table(connection: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }

    assert {
        "features",
        "scores",
        "explanations",
        "decisions",
        "assessment_runs",
        "assessment_decision_links",
    }.issubset(tables)


def test_assessment_persistence_is_complete(connection: sqlite3.Connection) -> None:
    stored = persist_assessment(connection, _assessment())

    assert stored.feature_id > 0
    assert stored.score_id > 0
    assert stored.explanation_id > 0
    assert stored.assessment_id > 0
    assert connection.execute("SELECT COUNT(*) FROM features").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM explanations").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM assessment_runs").fetchone()[0] == 1


def test_assessment_audit_context_is_persisted(connection: sqlite3.Connection) -> None:
    stored = persist_assessment(
        connection,
        _assessment(),
        AssessmentAuditContext(
            source_key=" csv ",
            processed_count=12,
            valid_count=10,
            rejected_count=2,
            warning_count=1,
        ),
    )

    row = connection.execute(
        """
        SELECT source_key, processed_count, valid_count, rejected_count, warning_count
        FROM assessment_runs WHERE assessment_id = ?
        """,
        (stored.assessment_id,),
    ).fetchone()
    assert row == ("csv", 12, 10, 2, 1)


def test_inconsistent_audit_counts_are_rejected(
    connection: sqlite3.Connection,
) -> None:
    with pytest.raises(ValueError, match="must equal"):
        persist_assessment(
            connection,
            _assessment(),
            AssessmentAuditContext(
                processed_count=3,
                valid_count=2,
                rejected_count=0,
            ),
        )

    assert connection.execute("SELECT COUNT(*) FROM assessment_runs").fetchone()[0] == 0


def test_feature_score_version_and_explanation_payloads_are_persisted(
    connection: sqlite3.Connection,
) -> None:
    persist_assessment(connection, _assessment())

    feature_payload = json.loads(
        connection.execute("SELECT feature_payload FROM features").fetchone()[0]
    )
    score_row = connection.execute(
        "SELECT applicant_id, risk_score, model_version FROM scores"
    ).fetchone()
    explanation_row = connection.execute(
        """
        SELECT explanation_payload, narrative_language, narrative_payload
        FROM explanations
        """
    ).fetchone()

    assert set(feature_payload) == set(FEATURE_COLUMNS)
    assert score_row == ("APP_0001", 0.42, "xgb_test_v1")
    explanation_payload = json.loads(explanation_row[0])
    narrative_payload = json.loads(explanation_row[2])
    assert explanation_payload["output_space"] == "raw_margin_log_odds"
    assert explanation_payload["contributions"][0]["feature_name"] == "low_balance_rate"
    assert explanation_row[1] == "en"
    assert "summary" in narrative_payload
    assert "disclaimer" in narrative_payload


def test_partial_failure_rolls_back_entire_assessment(
    connection: sqlite3.Connection,
) -> None:
    connection.executescript(
        """
        CREATE TRIGGER force_score_failure
        BEFORE INSERT ON scores
        BEGIN
            SELECT RAISE(ABORT, 'forced score failure');
        END;
        """
    )
    connection.commit()

    with pytest.raises(AssessmentPersistenceError, match="Could not persist"):
        persist_assessment(connection, _assessment())

    assert connection.execute("SELECT COUNT(*) FROM features").fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM explanations").fetchone()[0] == 0


@pytest.mark.parametrize("decision", list(HumanDecision))
def test_human_decision_values_are_persisted(
    connection: sqlite3.Connection, decision: HumanDecision
) -> None:
    decision_id = record_human_decision(
        connection,
        "APP_0001",
        decision,
        rationale="  Reviewed by loan officer.  ",
    )

    row = connection.execute(
        "SELECT decision_label, rationale FROM decisions WHERE decision_id = ?",
        (decision_id,),
    ).fetchone()
    assert row == (decision.value, "Reviewed by loan officer.")


def test_invalid_human_decision_is_rejected(connection: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="Allowed values"):
        record_human_decision(connection, "APP_0001", "AUTO_APPROVE")

    assert connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0


def test_human_decision_can_be_linked_to_assessment(
    connection: sqlite3.Connection,
) -> None:
    stored = persist_assessment(connection, _assessment())

    decision_id = record_human_decision(
        connection,
        "APP_0001",
        HumanDecision.REVIEW,
        assessment_id=stored.assessment_id,
    )

    link = connection.execute(
        "SELECT assessment_id, decision_id FROM assessment_decision_links"
    ).fetchone()
    assert link == (stored.assessment_id, decision_id)
    with pytest.raises(ValueError, match="already has"):
        record_human_decision(
            connection,
            "APP_0001",
            HumanDecision.APPROVE,
            assessment_id=stored.assessment_id,
        )
    assert connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1


def test_linked_decision_requires_matching_applicant(
    connection: sqlite3.Connection,
) -> None:
    stored = persist_assessment(connection, _assessment())

    with pytest.raises(ValueError, match="does not belong"):
        record_human_decision(
            connection,
            "APP_OTHER",
            HumanDecision.DECLINE,
            assessment_id=stored.assessment_id,
        )

    assert connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0


def test_model_assessment_does_not_create_human_decision(
    connection: sqlite3.Connection,
) -> None:
    persist_assessment(connection, _assessment())

    assert connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0


def test_database_path_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    environment_path = tmp_path / "environment.db"
    explicit_path = tmp_path / "explicit.db"
    monkeypatch.setenv("DB_PATH", str(environment_path))

    assert resolve_db_path(explicit_path) == explicit_path
    assert resolve_db_path() == environment_path
