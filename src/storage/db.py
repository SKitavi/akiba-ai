"""SQLite connection and schema bootstrap helpers.

Purpose: Create local DB connection and initialize MVP schema tables.
Owner: Sharon (ML Engineer).
Sprint day due: Day 3 (Aug 12) - parsing/features/storage milestone.
"""

import sqlite3
from pathlib import Path


# TODO(Sharon): Wire this to migrations + app startup path configuration.
def get_connection(db_path: Path) -> sqlite3.Connection:
    """Return a SQLite connection for local offline storage."""
    return sqlite3.connect(db_path)


# TODO(Sharon): Execute schema.sql and handle idempotent migration tracking.
def initialize_schema(connection: sqlite3.Connection, schema_path: Path) -> None:
    """Initialize SQLite schema for features, scores, and decisions tables."""
    raise NotImplementedError("Schema initialization is planned for Day 3.")
