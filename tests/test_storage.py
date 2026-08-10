"""Tests for storage scaffolding."""

import sqlite3
from pathlib import Path

import pytest

from src.storage.db import get_connection, initialize_schema


def test_get_connection_returns_sqlite_connection(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "test.db")
    try:
        assert isinstance(conn, sqlite3.Connection)
    finally:
        conn.close()


def test_initialize_schema_stub_raises_not_implemented(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "test.db")
    try:
        with pytest.raises(NotImplementedError):
            initialize_schema(conn, tmp_path / "schema.sql")
    finally:
        conn.close()
