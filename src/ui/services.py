"""Thin UI adapters around the frozen AkibaAI backend APIs."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
import random
import re
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import streamlit as st

from src.data_gen.generate_synthetic_data import (
    calibrate_default_labels,
    generate_dataset,
)
from src.features.build_features import FEATURE_COLUMNS, build_feature_table
from src.ingestion.normalization import NormalizationResult
from src.ingestion.ocr_parser import extract_text_from_image, parse_transaction_text
from src.ingestion.sms_parser import parse_sms_message
from src.application.assessment import AssessmentResult, assess_applicant
from src.model.loader import ModelBundle, load_model_bundle, resolve_model_path
from src.storage.assessment_store import (
    HumanDecision,
    PersistedAssessment,
    persist_assessment,
    record_human_decision,
)
from src.storage.db import get_connection, initialize_schema
from src.xai.narratives import generate_risk_narrative


_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "storage" / "schema.sql"


class UIInputError(ValueError):
    """Raised when user-provided input cannot enter the backend workflow."""


@dataclass(frozen=True)
class DemoDataset:
    """Deterministic synthetic applicant and transaction frames for demonstrations."""

    applicants: pd.DataFrame
    transactions: pd.DataFrame


@st.cache_data(show_spinner=False)
def load_demo_dataset(applicant_count: int = 30) -> DemoDataset:
    """Generate the existing synthetic dataset deterministically and cache it."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.seed(42)
        np.random.seed(42)
        applicants, transactions = generate_dataset(applicant_count)
        labeled_applicants = calibrate_default_labels(applicants, transactions)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    return DemoDataset(
        applicants=labeled_applicants,
        transactions=transactions,
    )


def demo_transactions_for_applicant(
    dataset: DemoDataset,
    applicant_id: str,
) -> pd.DataFrame:
    """Return one synthetic applicant's structured records without fabricating data."""
    records = dataset.transactions.loc[
        dataset.transactions["applicant_id"] == applicant_id
    ].copy()
    if records.empty:
        raise UIInputError(
            f"No synthetic transactions are available for applicant '{applicant_id}'."
        )
    return records


def read_csv_records(contents: bytes) -> pd.DataFrame:
    """Decode a CSV upload for canonical backend normalization."""
    if not contents:
        raise UIInputError("The uploaded CSV file is empty.")
    try:
        frame = pd.read_csv(BytesIO(contents))
    except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise UIInputError(
            "The CSV could not be read. Export it as a UTF-8 comma-separated file."
        ) from exc
    if frame.empty:
        raise UIInputError("The uploaded CSV contains no transaction rows.")
    return frame


def parse_sms_records(text: str) -> list[dict[str, object]]:
    """Parse provider SMS messages separated by one or more blank lines."""
    messages = [item.strip() for item in re.split(r"\n\s*\n+", text) if item.strip()]
    if not messages:
        raise UIInputError("Paste at least one supported provider SMS message.")
    return [parse_sms_message(message) for message in messages]


def parse_receipt_upload(contents: bytes, file_name: str) -> dict[str, object]:
    """Run the real OCR path for an uploaded receipt and return parsed fields."""
    if not contents:
        raise UIInputError("The uploaded receipt image is empty.")
    suffix = Path(file_name).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise UIInputError("Upload a PNG or JPEG receipt image.")

    with TemporaryDirectory(prefix="akiba_receipt_") as temporary_directory:
        image_path = Path(temporary_directory) / f"receipt{suffix}"
        image_path.write_bytes(contents)
        extracted_text = extract_text_from_image(image_path)
    return parse_transaction_text(extracted_text)


def build_feature_preview(
    normalization: NormalizationResult,
    applicant_id: str,
    applicants_df: pd.DataFrame | None,
) -> pd.Series:
    """Call canonical feature engineering and select one applicant preview row."""
    feature_table = build_feature_table(
        normalization.to_dataframe(),
        applicants_df=applicants_df,
    )
    selected = feature_table.loc[feature_table["applicant_id"] == applicant_id]
    if len(selected) != 1:
        raise UIInputError(
            f"Expected one financial summary for '{applicant_id}', found {len(selected)}."
        )
    row = selected.iloc[0]
    if any(feature not in row.index for feature in FEATURE_COLUMNS):
        raise UIInputError("The financial summary is missing required model inputs.")
    return row


@st.cache_resource(show_spinner=False)
def load_cached_model_bundle(model_path: str | None = None) -> ModelBundle:
    """Load and validate the local model once per Streamlit server process."""
    return load_model_bundle(model_path)


def run_assessment(
    normalization: NormalizationResult,
    applicant_id: str,
    applicants_df: pd.DataFrame | None,
    language: str = "en",
) -> AssessmentResult:
    """Run the canonical assessment service with a cached model bundle."""
    model_bundle = load_cached_model_bundle(str(resolve_model_path()))
    return assess_applicant(
        normalization.valid_transactions,
        model_bundle,
        applicant_id=applicant_id,
        applicants_df=applicants_df,
        language=language,
    )


def localize_assessment(
    assessment: AssessmentResult,
    language: str,
) -> AssessmentResult:
    """Regenerate only the backend-owned narrative in the requested language."""
    narrative = generate_risk_narrative(assessment.explanation, language=language)
    return replace(assessment, narrative=narrative)


def save_assessment(assessment: AssessmentResult) -> PersistedAssessment:
    """Initialize local storage and persist one assessment atomically."""
    with closing(get_connection()) as connection:
        initialize_schema(connection, _SCHEMA_PATH)
        return persist_assessment(connection, assessment)


def save_human_decision(
    applicant_id: str,
    decision: HumanDecision | str,
    rationale: str | None,
) -> int:
    """Persist an officer's decision through the backend decision boundary."""
    with closing(get_connection()) as connection:
        initialize_schema(connection, _SCHEMA_PATH)
        return record_human_decision(
            connection,
            applicant_id,
            decision,
            rationale,
        )
