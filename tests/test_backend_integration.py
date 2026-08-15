"""Golden-path integration test for the complete offline AkibaAI backend."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np

from src.application.assessment import assess_applicant
from src.data_gen.generate_synthetic_data import (
    calibrate_default_labels,
    generate_dataset,
)
from src.features.build_features import build_feature_table
from src.ingestion.normalization import normalize_transactions
from src.model.loader import load_model_bundle
from src.model.train import train_model
from src.storage.assessment_store import (
    HumanDecision,
    persist_assessment,
    record_human_decision,
)
from src.storage.db import get_connection, initialize_schema
from src.xai.narratives import NarrativeLanguage


_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "src" / "storage" / "schema.sql"


def test_structured_synthetic_backend_workflow(tmp_path: Path) -> None:
    """Exercise normalization through persistence without importing Streamlit."""
    random.seed(42)
    np.random.seed(42)
    applicants, structured_transactions = generate_dataset(30)
    labeled_applicants = calibrate_default_labels(applicants, structured_transactions)

    normalization = normalize_transactions(structured_transactions)
    assert normalization.rejected_count == 0
    assert normalization.valid_count == len(structured_transactions)

    feature_table = build_feature_table(
        normalization.to_dataframe(), applicants_df=labeled_applicants
    )
    training_table = feature_table.merge(
        labeled_applicants[["applicant_id", "default_label"]], on="applicant_id"
    )
    model_path = tmp_path / "backend_model.json"
    train_model(training_table, model_path)
    model_bundle = load_model_bundle(model_path)

    applicant_id = str(feature_table.iloc[0]["applicant_id"])
    applicant_transactions = tuple(
        transaction
        for transaction in normalization.valid_transactions
        if transaction.applicant_id == applicant_id
    )
    english = assess_applicant(
        applicant_transactions,
        model_bundle,
        applicant_id=applicant_id,
        applicants_df=labeled_applicants,
        language="en",
        top_n=3,
    )
    kiswahili = assess_applicant(
        applicant_transactions,
        model_bundle,
        applicant_id=applicant_id,
        applicants_df=labeled_applicants,
        language="sw",
        top_n=3,
    )

    assert english.model_version == "xgb_v1"
    assert 0.0 <= english.risk_score <= 1.0
    assert english.narrative.language is NarrativeLanguage.ENGLISH
    assert kiswahili.narrative.language is NarrativeLanguage.KISWAHILI
    assert kiswahili.risk_score == english.risk_score
    assert english.explanation.contributions

    connection = get_connection(tmp_path / "backend.db")
    try:
        initialize_schema(connection, _SCHEMA_PATH)
        stored = persist_assessment(connection, english)

        assert stored.feature_id > 0
        assert connection.execute("SELECT COUNT(*) FROM features").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 1
        assert (
            connection.execute("SELECT COUNT(*) FROM explanations").fetchone()[0] == 1
        )
        assert connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0

        record_human_decision(
            connection,
            applicant_id,
            HumanDecision.REVIEW,
            rationale="Human reviewer requested supporting documents.",
        )
        decision = connection.execute(
            "SELECT decision_label FROM decisions"
        ).fetchone()[0]
        assert decision == "REVIEW"
    finally:
        connection.close()
