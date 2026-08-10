-- Purpose: MVP storage schema for offline credit scoring simulation.
-- Owner: Sharon (ML Engineer).
-- Sprint day due: Day 3 (Aug 12) - parsing/features/storage milestone.

CREATE TABLE IF NOT EXISTS features (
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    applicant_id TEXT NOT NULL,
    feature_payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scores (
    score_id INTEGER PRIMARY KEY AUTOINCREMENT,
    applicant_id TEXT NOT NULL,
    risk_score REAL NOT NULL,
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    applicant_id TEXT NOT NULL,
    decision_label TEXT NOT NULL,
    rationale TEXT,
    created_at TEXT NOT NULL
);
