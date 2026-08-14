"""Tests for the SQLite storage layer (db.py)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from src.storage.db import (
    get_connection,
    initialize_schema,
    save_features,
    save_score,
    save_decision,
)

# Path to the canonical schema SQL file
_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "src" / "storage" / "schema.sql"


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def test_get_connection_returns_sqlite_connection(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "test.db")
    try:
        assert isinstance(conn, sqlite3.Connection)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def test_initialize_schema_creates_tables(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "test.db")
    try:
        initialize_schema(conn, _SCHEMA_PATH)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert {"features", "scores", "decisions"}.issubset(tables)
    finally:
        conn.close()


def test_initialize_schema_is_idempotent(tmp_path: Path) -> None:
    """Calling initialize_schema twice must not raise an error."""
    conn = get_connection(tmp_path / "test.db")
    try:
        initialize_schema(conn, _SCHEMA_PATH)
        initialize_schema(conn, _SCHEMA_PATH)  # second call — must be safe
    finally:
        conn.close()


def test_initialize_schema_missing_file_raises(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "test.db")
    try:
        with pytest.raises(FileNotFoundError):
            initialize_schema(conn, tmp_path / "nonexistent.sql")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Feature persistence
# ---------------------------------------------------------------------------

def test_save_features_inserts_row(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "test.db")
    try:
        initialize_schema(conn, _SCHEMA_PATH)
        feature_dict = {"tx_per_month": 12.5, "net_flow_ratio": 0.35}
        save_features(conn, "APP_0001", feature_dict)

        cursor = conn.execute(
            "SELECT applicant_id, feature_payload FROM features WHERE applicant_id = ?",
            ("APP_0001",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "APP_0001"
        stored = json.loads(row[1])
        assert stored["tx_per_month"] == 12.5
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Score persistence
# ---------------------------------------------------------------------------

def test_save_score_inserts_row(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "test.db")
    try:
        initialize_schema(conn, _SCHEMA_PATH)
        save_score(conn, "APP_0002", 0.73, "xgb_v1")

        cursor = conn.execute(
            "SELECT applicant_id, risk_score, model_version FROM scores WHERE applicant_id = ?",
            ("APP_0002",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "APP_0002"
        assert abs(row[1] - 0.73) < 1e-6
        assert row[2] == "xgb_v1"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Decision persistence
# ---------------------------------------------------------------------------

def test_save_decision_inserts_row(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "test.db")
    try:
        initialize_schema(conn, _SCHEMA_PATH)
        save_decision(conn, "APP_0003", "APPROVE", "Net flow positive for 5/6 months.")

        cursor = conn.execute(
            "SELECT applicant_id, decision_label, rationale FROM decisions WHERE applicant_id = ?",
            ("APP_0003",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[1] == "APPROVE"
        assert "positive" in row[2]
    finally:
        conn.close()


def test_save_decision_without_rationale(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "test.db")
    try:
        initialize_schema(conn, _SCHEMA_PATH)
        save_decision(conn, "APP_0004", "DECLINE")

        cursor = conn.execute(
            "SELECT rationale FROM decisions WHERE applicant_id = ?",
            ("APP_0004",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] is None
    finally:
        conn.close()
