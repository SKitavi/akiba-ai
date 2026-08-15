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

CREATE TABLE IF NOT EXISTS explanations (
    explanation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    applicant_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    explanation_payload TEXT NOT NULL,
    narrative_language TEXT NOT NULL,
    narrative_payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    applicant_id TEXT NOT NULL,
    decision_label TEXT NOT NULL,
    rationale TEXT,
    created_at TEXT NOT NULL
);

-- Links the independently stored artifacts that make up one assessment run.
-- This additive table keeps existing databases migration-compatible.
CREATE TABLE IF NOT EXISTS assessment_runs (
    assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    applicant_id TEXT NOT NULL,
    feature_id INTEGER NOT NULL UNIQUE,
    score_id INTEGER NOT NULL UNIQUE,
    explanation_id INTEGER NOT NULL UNIQUE,
    source_key TEXT,
    processed_count INTEGER,
    valid_count INTEGER,
    rejected_count INTEGER,
    warning_count INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (feature_id) REFERENCES features(feature_id),
    FOREIGN KEY (score_id) REFERENCES scores(score_id),
    FOREIGN KEY (explanation_id) REFERENCES explanations(explanation_id)
);

-- Decisions remain human-owned records; this table associates at most one
-- decision with the assessment the officer reviewed.
CREATE TABLE IF NOT EXISTS assessment_decision_links (
    assessment_id INTEGER PRIMARY KEY,
    decision_id INTEGER NOT NULL UNIQUE,
    FOREIGN KEY (assessment_id) REFERENCES assessment_runs(assessment_id),
    FOREIGN KEY (decision_id) REFERENCES decisions(decision_id)
);

CREATE INDEX IF NOT EXISTS idx_assessment_runs_created_at
    ON assessment_runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_assessment_runs_applicant_id
    ON assessment_runs(applicant_id);
