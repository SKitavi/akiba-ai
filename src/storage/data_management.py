"""Transactional management operations for locally persisted assessment data."""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3


class AssessmentDataResetError(RuntimeError):
    """Raised when assessment data cannot be cleared atomically."""


@dataclass(frozen=True)
class AssessmentDataCounts:
    """Row counts for every assessment-owned storage table."""

    features: int
    scores: int
    explanations: int
    decisions: int
    assessment_runs: int
    decision_links: int


_COUNT_QUERIES = {
    "features": "SELECT COUNT(*) FROM features",
    "scores": "SELECT COUNT(*) FROM scores",
    "explanations": "SELECT COUNT(*) FROM explanations",
    "decisions": "SELECT COUNT(*) FROM decisions",
    "assessment_runs": "SELECT COUNT(*) FROM assessment_runs",
    "decision_links": "SELECT COUNT(*) FROM assessment_decision_links",
}


def get_assessment_data_counts(
    connection: sqlite3.Connection,
) -> AssessmentDataCounts:
    """Return current row counts without changing storage."""
    values = {
        name: int(connection.execute(query).fetchone()[0])
        for name, query in _COUNT_QUERIES.items()
    }
    return AssessmentDataCounts(**values)


def reset_assessment_data(connection: sqlite3.Connection) -> AssessmentDataCounts:
    """Delete all assessment-owned rows atomically and return deleted counts."""
    counts = get_assessment_data_counts(connection)
    try:
        with connection:
            for table in (
                "assessment_decision_links",
                "assessment_runs",
                "decisions",
                "explanations",
                "scores",
                "features",
            ):
                connection.execute(f"DELETE FROM {table}")
            connection.execute(
                """
                DELETE FROM sqlite_sequence
                WHERE name IN (
                    'assessment_runs', 'decisions', 'explanations',
                    'scores', 'features'
                )
                """
            )
    except sqlite3.DatabaseError as exc:
        raise AssessmentDataResetError(
            "Could not reset persisted assessment data."
        ) from exc
    return counts
