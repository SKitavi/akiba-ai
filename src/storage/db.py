"""SQLite connection and schema bootstrap helpers.

Purpose: Create local DB connection and initialize MVP schema tables.
Owner: Sharon (ML Engineer).
Sprint day due: Day 3 (Aug 12) - parsing/features/storage milestone.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Return a SQLite connection for local offline storage."""
    return sqlite3.connect(db_path)


def initialize_schema(connection: sqlite3.Connection, schema_path: Path) -> None:
    """Initialize SQLite schema for features, scores, and decisions tables.

    Reads and executes the SQL DDL from ``schema_path`` using executescript so
    all ``CREATE TABLE IF NOT EXISTS`` statements are applied idempotently.

    Args:
        connection:  An open ``sqlite3.Connection`` instance.
        schema_path: Path to the ``.sql`` file containing DDL statements.

    Raises:
        FileNotFoundError: If ``schema_path`` does not exist.
    """
    schema_path = Path(schema_path)
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    sql = schema_path.read_text(encoding="utf-8")
    connection.executescript(sql)
    connection.commit()


# ---------------------------------------------------------------------------
# Feature persistence helpers
# ---------------------------------------------------------------------------

def save_features(
    connection: sqlite3.Connection,
    applicant_id: str,
    feature_dict: dict[str, Any],
) -> None:
    """Persist a feature payload for one applicant as a JSON blob.

    Args:
        connection:   Open SQLite connection with the schema already initialised.
        applicant_id: Applicant identifier string (e.g. ``'APP_0001'``).
        feature_dict: Dictionary of feature name → value produced by
                      ``build_feature_table``.
    """
    payload = json.dumps(feature_dict, default=float)
    created_at = datetime.now(timezone.utc).isoformat()
    connection.execute(
        "INSERT INTO features (applicant_id, feature_payload, created_at) VALUES (?, ?, ?)",
        (applicant_id, payload, created_at),
    )
    connection.commit()


def save_score(
    connection: sqlite3.Connection,
    applicant_id: str,
    risk_score: float,
    model_version: str,
) -> None:
    """Persist a risk score for one applicant.

    Args:
        connection:    Open SQLite connection with the schema already initialised.
        applicant_id:  Applicant identifier string.
        risk_score:    Calibrated probability of default in ``[0, 1]``.
        model_version: Model version tag (e.g. ``'xgb_v1'``).
    """
    created_at = datetime.now(timezone.utc).isoformat()
    connection.execute(
        "INSERT INTO scores (applicant_id, risk_score, model_version, created_at) VALUES (?, ?, ?, ?)",
        (applicant_id, risk_score, model_version, created_at),
    )
    connection.commit()


def save_decision(
    connection: sqlite3.Connection,
    applicant_id: str,
    decision_label: str,
    rationale: Optional[str] = None,
) -> None:
    """Persist a credit decision for one applicant.

    Args:
        connection:     Open SQLite connection with the schema already initialised.
        applicant_id:   Applicant identifier string.
        decision_label: Human-readable decision label, e.g. ``'APPROVE'`` or
                        ``'DECLINE'``.
        rationale:      Optional free-text explanation (e.g. top SHAP driver).
    """
    created_at = datetime.now(timezone.utc).isoformat()
    connection.execute(
        "INSERT INTO decisions (applicant_id, decision_label, rationale, created_at) VALUES (?, ?, ?, ?)",
        (applicant_id, decision_label, rationale, created_at),
    )
    connection.commit()
