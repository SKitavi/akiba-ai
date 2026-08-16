"""Tests for transactional assessment-data management."""

from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from src.storage.data_management import (
    AssessmentDataCounts,
    AssessmentDataResetError,
    get_assessment_data_counts,
    reset_assessment_data,
)
from src.storage.db import get_connection, initialize_schema
from src.storage.seed_dashboard_demo import seed_dashboard_assessments


_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "src" / "storage" / "schema.sql"


@pytest.fixture()
def connection(tmp_path: Path) -> sqlite3.Connection:
    database = get_connection(tmp_path / "data-management.db")
    initialize_schema(database, _SCHEMA_PATH)
    yield database
    database.close()


def test_reset_removes_all_assessment_data_and_resets_ids(
    connection: sqlite3.Connection,
) -> None:
    seed_dashboard_assessments(connection, count=4)

    deleted = reset_assessment_data(connection)

    assert deleted == AssessmentDataCounts(4, 4, 4, 3, 4, 3)
    assert get_assessment_data_counts(connection) == AssessmentDataCounts(
        0, 0, 0, 0, 0, 0
    )
    seed_dashboard_assessments(connection, count=1)
    assert (
        connection.execute("SELECT assessment_id FROM assessment_runs").fetchone()[0]
        == 1
    )


def test_reset_failure_rolls_back_every_table(
    connection: sqlite3.Connection,
) -> None:
    seed_dashboard_assessments(connection, count=4)
    original = get_assessment_data_counts(connection)
    connection.executescript(
        """
        CREATE TRIGGER prevent_score_reset
        BEFORE DELETE ON scores
        BEGIN
            SELECT RAISE(ABORT, 'forced reset failure');
        END;
        """
    )
    connection.commit()

    with pytest.raises(AssessmentDataResetError, match="Could not reset"):
        reset_assessment_data(connection)

    assert get_assessment_data_counts(connection) == original
