"""Typed read models and queries for persisted assessment analytics."""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3


@dataclass(frozen=True)
class AssessmentHistoryItem:
    """One persisted assessment with its optional linked officer decision."""

    assessment_id: int
    applicant_id: str
    assessed_at: str
    model_version: str
    risk_score: float
    source_key: str | None
    processed_count: int | None
    valid_count: int | None
    rejected_count: int | None
    warning_count: int | None
    decision: str | None
    rationale: str | None
    decided_at: str | None


@dataclass(frozen=True)
class ScoreRangeCount:
    """Count within a neutral, fixed-width model score interval."""

    label: str
    count: int


@dataclass(frozen=True)
class AssessmentAnalytics:
    """Aggregate operational facts derived from persisted assessment runs."""

    total_assessments: int
    awaiting_decision: int
    recorded_decisions: int
    decision_counts: dict[str, int]
    source_counts: dict[str, int]
    assessments_with_audit: int
    processed_records: int
    valid_records: int
    rejected_records: int
    warning_count: int
    score_distribution: tuple[ScoreRangeCount, ...]


_HISTORY_QUERY = """
    SELECT
        ar.assessment_id,
        ar.applicant_id,
        ar.created_at,
        s.model_version,
        s.risk_score,
        ar.source_key,
        ar.processed_count,
        ar.valid_count,
        ar.rejected_count,
        ar.warning_count,
        d.decision_label,
        d.rationale,
        d.created_at
    FROM assessment_runs AS ar
    JOIN scores AS s ON s.score_id = ar.score_id
    LEFT JOIN assessment_decision_links AS adl
        ON adl.assessment_id = ar.assessment_id
    LEFT JOIN decisions AS d ON d.decision_id = adl.decision_id
"""


def list_assessment_history(
    connection: sqlite3.Connection,
    *,
    limit: int = 100,
    applicant_query: str | None = None,
) -> tuple[AssessmentHistoryItem, ...]:
    """Return newest linked assessment runs, optionally filtered by applicant."""
    if not isinstance(limit, int) or limit <= 0 or limit > 1000:
        raise ValueError("limit must be an integer between 1 and 1000.")
    query = _HISTORY_QUERY
    parameters: list[object] = []
    normalized_query = applicant_query.strip() if applicant_query else ""
    if normalized_query:
        query += " WHERE LOWER(ar.applicant_id) LIKE LOWER(?)"
        parameters.append(f"%{normalized_query}%")
    query += " ORDER BY ar.created_at DESC, ar.assessment_id DESC LIMIT ?"
    parameters.append(limit)
    rows = connection.execute(query, parameters).fetchall()
    return tuple(AssessmentHistoryItem(*row) for row in rows)


def get_assessment_analytics(connection: sqlite3.Connection) -> AssessmentAnalytics:
    """Aggregate assessment, decision, ingestion, and score-distribution facts."""
    summary = connection.execute(
        """
        SELECT
            COUNT(*),
            SUM(CASE WHEN adl.decision_id IS NULL THEN 1 ELSE 0 END),
            COUNT(adl.decision_id),
            SUM(CASE WHEN ar.processed_count IS NOT NULL THEN 1 ELSE 0 END),
            COALESCE(SUM(ar.processed_count), 0),
            COALESCE(SUM(ar.valid_count), 0),
            COALESCE(SUM(ar.rejected_count), 0),
            COALESCE(SUM(ar.warning_count), 0)
        FROM assessment_runs AS ar
        LEFT JOIN assessment_decision_links AS adl
            ON adl.assessment_id = ar.assessment_id
        """
    ).fetchone()
    decision_counts = {label: 0 for label in ("APPROVE", "REVIEW", "DECLINE")}
    decision_counts.update(
        dict(
            connection.execute(
                """
                SELECT d.decision_label, COUNT(*)
                FROM assessment_decision_links AS adl
                JOIN decisions AS d ON d.decision_id = adl.decision_id
                GROUP BY d.decision_label
                """
            ).fetchall()
        )
    )
    source_counts = dict(
        connection.execute(
            """
            SELECT COALESCE(source_key, 'unspecified'), COUNT(*)
            FROM assessment_runs
            GROUP BY COALESCE(source_key, 'unspecified')
            ORDER BY COUNT(*) DESC, COALESCE(source_key, 'unspecified')
            """
        ).fetchall()
    )
    score_counts = [0, 0, 0, 0]
    for score, count in connection.execute(
        """
        SELECT
            CASE
                WHEN s.risk_score < 0.25 THEN 0
                WHEN s.risk_score < 0.50 THEN 1
                WHEN s.risk_score < 0.75 THEN 2
                ELSE 3
            END,
            COUNT(*)
        FROM assessment_runs AS ar
        JOIN scores AS s ON s.score_id = ar.score_id
        GROUP BY 1
        """
    ).fetchall():
        score_counts[int(score)] = int(count)
    labels = ("0.00–0.24", "0.25–0.49", "0.50–0.74", "0.75–1.00")
    return AssessmentAnalytics(
        total_assessments=int(summary[0]),
        awaiting_decision=int(summary[1] or 0),
        recorded_decisions=int(summary[2]),
        assessments_with_audit=int(summary[3] or 0),
        processed_records=int(summary[4]),
        valid_records=int(summary[5]),
        rejected_records=int(summary[6]),
        warning_count=int(summary[7]),
        decision_counts=decision_counts,
        source_counts=source_counts,
        score_distribution=tuple(
            ScoreRangeCount(label, count) for label, count in zip(labels, score_counts)
        ),
    )
