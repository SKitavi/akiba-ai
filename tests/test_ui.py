"""Streamlit workflow tests for the AkibaAI officer workstation."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
import xgboost as xgb

from src.features.build_features import FEATURE_COLUMNS
from src.ingestion.normalization import TransactionContext, normalize_transactions
from src.ui.services import UIInputError, parse_sms_records, read_csv_records


_APP_PATH = Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py"


@pytest.fixture(scope="module")
def ui_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create one small canonical model artifact for UI integration checks."""
    artifact_directory = tmp_path_factory.mktemp("ui_model")
    rng = np.random.default_rng(21)
    features = pd.DataFrame(
        {feature: rng.normal(size=60) for feature in FEATURE_COLUMNS}
    )
    labels = np.array([0, 1] * 30)
    model = xgb.XGBClassifier(
        n_estimators=8,
        max_depth=2,
        random_state=42,
        n_jobs=1,
    ).fit(features, labels)
    model_path = artifact_directory / "ui_model.json"
    model.save_model(str(model_path))
    model_path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "model_version": "ui_test_v1",
                "n_features": len(FEATURE_COLUMNS),
                "feature_columns": FEATURE_COLUMNS,
            }
        ),
        encoding="utf-8",
    )
    return model_path


def _button(app: AppTest, label: str):
    return next(button for button in app.button if button.label == label)


def _start_demo_assessment(app: AppTest) -> AppTest:
    _button(app, "Start new assessment").click().run()
    _button(app, "Continue to transactions").click().run()
    _button(app, "Validate transactions").click().run()
    return app


def _has_markdown(app: AppTest, text: str) -> bool:
    return any(text.lower() in str(item.value).lower() for item in app.markdown)


def test_application_loads_and_default_navigation_works() -> None:
    app = AppTest.from_file(_APP_PATH, default_timeout=30).run()

    assert not app.exception
    assert [button.label for button in app.button[:3]] == [
        "Overview",
        "New Assessment",
        "History",
    ]
    assert _has_markdown(app, "Credit assessment workspace")

    _button(app, "History").click().run()
    assert not app.exception
    assert _has_markdown(app, "No session history")


def test_demo_validation_reaches_financial_summary() -> None:
    app = AppTest.from_file(_APP_PATH, default_timeout=30).run()
    _start_demo_assessment(app)

    result = app.session_state["normalization_result"]
    assert not app.exception
    assert result.valid_count > 0
    assert result.rejected_count == 0
    assert app.session_state["feature_preview"] is not None
    assert _has_markdown(app, "Financial summary")
    assert not _button(app, "Continue to assessment").disabled


def test_zero_valid_records_block_assessment() -> None:
    result = normalize_transactions(
        [{"provider": "unsupported"}],
        context=TransactionContext(applicant_id="CUSTOM-001"),
    )
    app = AppTest.from_file(_APP_PATH, default_timeout=30).run()
    app.session_state["route"] = "assessment"
    app.session_state["assessment_step"] = 2
    app.session_state["applicant_id"] = "CUSTOM-001"
    app.session_state["source_key"] = "csv"
    app.session_state["normalization_result"] = result
    app.session_state["feature_preview"] = None
    app.run()

    assert not app.exception
    assert _has_markdown(app, "No valid transactions are available")
    assert _button(app, "Continue to assessment").disabled
    assert len(app.dataframe) == 1


def test_model_missing_state_is_understandable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.json"))
    app = AppTest.from_file(_APP_PATH, default_timeout=30).run()
    _start_demo_assessment(app)
    _button(app, "Continue to assessment").click().run()

    assert not app.exception
    assert _has_markdown(app, "local scoring model is not available")
    assert any("src.model.run_training" in str(item.value) for item in app.code)


def test_complete_assessment_language_and_persistence_workflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ui_model_path: Path,
) -> None:
    database_path = tmp_path / "ui.db"
    monkeypatch.setenv("MODEL_PATH", str(ui_model_path))
    monkeypatch.setenv("DB_PATH", str(database_path))
    app = AppTest.from_file(_APP_PATH, default_timeout=60).run()
    _start_demo_assessment(app)
    _button(app, "Continue to assessment").click().run()
    _button(app, "Run model assessment").click().run(timeout=60)

    assert not app.exception
    assert _has_markdown(app, "Model risk score")
    assert _has_markdown(app, "Model version: ui_test_v1")
    assert _has_markdown(app, "Factors increasing estimated risk")
    assert _has_markdown(app, "Factors reducing estimated risk")
    assert not _has_markdown(app, "Low Risk")
    assert app.session_state["assessment_result"].narrative.language.value == "en"

    next(
        radio for radio in app.radio if radio.label == "Explanation language"
    ).set_value("Kiswahili").run()
    assert app.session_state["assessment_result"].narrative.language.value == "sw"
    assert "alama ya hatari" in app.session_state["assessment_result"].narrative.summary

    _button(app, "Save assessment").click().run()
    app.run()
    assert app.session_state["assessment_saved"] is True
    decision = next(radio for radio in app.radio if radio.label == "Decision")
    assert decision.value is None

    _button(app, "Record decision").click().run()
    assert _has_markdown(app, "Select an officer decision")

    with sqlite3.connect(database_path) as connection:
        counts = [
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("features", "scores", "explanations", "decisions")
        ]
    assert counts == [1, 1, 1, 0]

    next(radio for radio in app.radio if radio.label == "Decision").set_value(
        "REVIEW"
    ).run()
    next(area for area in app.text_area if area.label == "Rationale").input(
        "Supervisor review requested."
    ).run()
    _button(app, "Record decision").click().run()
    app.run()

    with sqlite3.connect(database_path) as connection:
        counts = [
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("features", "scores", "explanations", "decisions")
        ]
    assert counts == [1, 1, 1, 1]
    assert app.session_state["decision_saved"] is True
    assert app.session_state["session_history"][0]["decision"] == "Review"


def test_upload_adapters_reject_empty_input() -> None:
    with pytest.raises(UIInputError, match="empty"):
        read_csv_records(b"")
    with pytest.raises(UIInputError, match="at least one"):
        parse_sms_records("  \n")
