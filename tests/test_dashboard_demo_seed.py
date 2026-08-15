"""Tests for idempotent synthetic dashboard assessment seeding."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.storage.analytics import get_assessment_analytics, list_assessment_history
from src.storage.db import get_connection, initialize_schema
from src.storage.seed_dashboard_demo import seed_dashboard_assessments


_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "src" / "storage" / "schema.sql"


def test_dashboard_seed_is_complete_and_idempotent(tmp_path: Path) -> None:
    connection = get_connection(tmp_path / "dashboard-demo.db")
    try:
        initialize_schema(connection, _SCHEMA_PATH)

        first = seed_dashboard_assessments(connection, count=12)
        second = seed_dashboard_assessments(connection, count=12)
        analytics = get_assessment_analytics(connection)
        history = list_assessment_history(connection)

        assert first.created_count == 12
        assert second.created_count == 0
        assert second.existing_count == 12
        assert analytics.total_assessments == 12
        assert analytics.recorded_decisions == 9
        assert analytics.awaiting_decision == 3
        assert analytics.source_counts == {"dashboard_demo": 12}
        assert [item.count for item in analytics.score_distribution] == [3, 3, 3, 3]
        assert len(history) == 12
        assert all(item.applicant_id.startswith("DASH_DEMO_") for item in history)
    finally:
        connection.close()


@pytest.mark.parametrize("count", [0, 101, True])
def test_dashboard_seed_rejects_invalid_count(
    tmp_path: Path,
    count: int,
) -> None:
    connection = get_connection(tmp_path / "invalid-dashboard-demo.db")
    try:
        initialize_schema(connection, _SCHEMA_PATH)
        with pytest.raises(ValueError, match="between 1 and 100"):
            seed_dashboard_assessments(connection, count=count)
    finally:
        connection.close()
