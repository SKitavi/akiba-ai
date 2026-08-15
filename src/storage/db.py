"""SQLite connection and schema bootstrap helpers.

Purpose: Create local DB connection and initialize MVP schema tables.
Owner: Sharon (ML Engineer).
Sprint day due: Day 3 (Aug 12) - parsing/features/storage milestone.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "akiba_ai.db"


def resolve_db_path(db_path: Path | str | None = None) -> Path:
    """Resolve database path by explicit argument, environment, then default."""
    if db_path is not None:
        return Path(db_path).expanduser()
    configured_path = os.environ.get("DB_PATH")
    if configured_path:
        return Path(configured_path).expanduser()
    return DEFAULT_DB_PATH


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Return a SQLite connection for local offline storage."""
    return sqlite3.connect(resolve_db_path(db_path))


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
    *,
    commit: bool = True,
) -> int:
    """Persist a feature payload for one applicant as a JSON blob.

    Args:
        connection:   Open SQLite connection with the schema already initialised.
        applicant_id: Applicant identifier string (e.g. ``'APP_0001'``).
        feature_dict: Dictionary of feature name → value produced by
                      ``build_feature_table``.
    """
    payload = json.dumps(feature_dict, default=float)
    created_at = datetime.now(timezone.utc).isoformat()
    cursor = connection.execute(
        "INSERT INTO features (applicant_id, feature_payload, created_at) VALUES (?, ?, ?)",
        (applicant_id, payload, created_at),
    )
    if commit:
        connection.commit()
    return int(cursor.lastrowid)


def save_score(
    connection: sqlite3.Connection,
    applicant_id: str,
    risk_score: float,
    model_version: str,
    *,
    commit: bool = True,
) -> int:
    """Persist a risk score for one applicant.

    Args:
        connection:    Open SQLite connection with the schema already initialised.
        applicant_id:  Applicant identifier string.
        risk_score:    XGBoost model risk score in ``[0, 1]``.
        model_version: Model version tag (e.g. ``'xgb_v1'``).
    """
    created_at = datetime.now(timezone.utc).isoformat()
    cursor = connection.execute(
        "INSERT INTO scores (applicant_id, risk_score, model_version, created_at) VALUES (?, ?, ?, ?)",
        (applicant_id, risk_score, model_version, created_at),
    )
    if commit:
        connection.commit()
    return int(cursor.lastrowid)


def save_explanation(
    connection: sqlite3.Connection,
    applicant_id: str,
    model_version: str,
    explanation_dict: dict[str, Any],
    narrative_language: str,
    narrative_dict: dict[str, Any],
    *,
    commit: bool = True,
) -> int:
    """Persist structured SHAP and localized narrative payloads."""
    created_at = datetime.now(timezone.utc).isoformat()
    cursor = connection.execute(
        """
        INSERT INTO explanations (
            applicant_id,
            model_version,
            explanation_payload,
            narrative_language,
            narrative_payload,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            applicant_id,
            model_version,
            json.dumps(explanation_dict, default=float),
            narrative_language,
            json.dumps(narrative_dict, default=float),
            created_at,
        ),
    )
    if commit:
        connection.commit()
    return int(cursor.lastrowid)


def save_decision(
    connection: sqlite3.Connection,
    applicant_id: str,
    decision_label: str,
    rationale: Optional[str] = None,
    *,
    commit: bool = True,
) -> int:
    """Persist a credit decision for one applicant.

    Args:
        connection:     Open SQLite connection with the schema already initialised.
        applicant_id:   Applicant identifier string.
        decision_label: Human-readable decision label, e.g. ``'APPROVE'`` or
                        ``'DECLINE'``.
        rationale:      Optional free-text explanation (e.g. top SHAP driver).
    """
    created_at = datetime.now(timezone.utc).isoformat()
    cursor = connection.execute(
        "INSERT INTO decisions (applicant_id, decision_label, rationale, created_at) VALUES (?, ?, ?, ?)",
        (applicant_id, decision_label, rationale, created_at),
    )
    if commit:
        connection.commit()
    return int(cursor.lastrowid)
