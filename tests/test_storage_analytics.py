"""Tests for persisted assessment history and aggregate analytics."""

from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from src.storage.analytics import get_assessment_analytics, list_assessment_history
from src.storage.assessment_store import (
    AssessmentAuditContext,
    HumanDecision,
    persist_assessment,
    record_human_decision,
)
from src.storage.db import get_connection, initialize_schema
from tests.test_assessment_store import _assessment


_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "src" / "storage" / "schema.sql"


@pytest.fixture()
def connection(tmp_path: Path) -> sqlite3.Connection:
    database = get_connection(tmp_path / "analytics.db")
    initialize_schema(database, _SCHEMA_PATH)
    yield database
    database.close()


def test_empty_analytics_are_zeroed(connection: sqlite3.Connection) -> None:
    analytics = get_assessment_analytics(connection)

    assert analytics.total_assessments == 0
    assert analytics.awaiting_decision == 0
    assert analytics.recorded_decisions == 0
    assert analytics.decision_counts == {"APPROVE": 0, "REVIEW": 0, "DECLINE": 0}
    assert sum(item.count for item in analytics.score_distribution) == 0
    assert list_assessment_history(connection) == ()


def test_analytics_report_linked_runs_and_ingestion_quality(
    connection: sqlite3.Connection,
) -> None:
    first = persist_assessment(
        connection,
        _assessment(),
        AssessmentAuditContext("demo", 10, 9, 1, 2),
    )
    persist_assessment(
        connection,
        _assessment(),
        AssessmentAuditContext("csv", 8, 8, 0, 0),
    )
    record_human_decision(
        connection,
        "APP_0001",
        HumanDecision.APPROVE,
        "Verified evidence.",
        assessment_id=first.assessment_id,
    )

    analytics = get_assessment_analytics(connection)

    assert analytics.total_assessments == 2
    assert analytics.awaiting_decision == 1
    assert analytics.recorded_decisions == 1
    assert analytics.decision_counts["APPROVE"] == 1
    assert analytics.source_counts == {"csv": 1, "demo": 1}
    assert analytics.assessments_with_audit == 2
    assert (analytics.processed_records, analytics.valid_records) == (18, 17)
    assert (analytics.rejected_records, analytics.warning_count) == (1, 2)
    assert sum(item.count for item in analytics.score_distribution) == 2

    history = list_assessment_history(connection)
    assert len(history) == 2
    linked = next(item for item in history if item.assessment_id == first.assessment_id)
    assert linked.decision == "APPROVE"
    assert linked.rationale == "Verified evidence."
    assert linked.source_key == "demo"


def test_history_filter_and_limit_are_enforced(connection: sqlite3.Connection) -> None:
    persist_assessment(connection, _assessment())
    persist_assessment(connection, _assessment())

    assert (
        len(list_assessment_history(connection, applicant_query="0001", limit=1)) == 1
    )
    assert list_assessment_history(connection, applicant_query="missing") == ()
    with pytest.raises(ValueError, match="between 1 and 1000"):
        list_assessment_history(connection, limit=0)
