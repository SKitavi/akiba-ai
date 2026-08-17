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
from src.storage.db import get_connection, initialize_schema
from src.storage.seed_dashboard_demo import seed_dashboard_assessments
from src.ui.services import UIInputError, parse_sms_records, read_csv_records


_APP_PATH = Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py"
_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "src" / "storage" / "schema.sql"


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


def test_application_loads_and_default_navigation_works(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DB_PATH", str(tmp_path / "navigation.db"))
    app = AppTest.from_file(_APP_PATH, default_timeout=30).run()

    assert not app.exception
    button_labels = {button.label for button in app.button}
    assert {"Overview", "New Assessment", "History", "Settings"} <= button_labels
    assert _has_markdown(app, "Loan assessment overview")

    _button(app, "History").click().run()
    assert not app.exception
    assert _has_markdown(app, "No persisted history")


def test_settings_can_reset_and_reload_demo_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "settings.db"
    monkeypatch.setenv("DB_PATH", str(database_path))
    connection = get_connection(database_path)
    try:
        initialize_schema(connection, _SCHEMA_PATH)
        seed_dashboard_assessments(connection, count=4)
    finally:
        connection.close()

    app = AppTest.from_file(_APP_PATH, default_timeout=30).run()
    _button(app, "Settings").click().run()

    assert not any(button.label == "Reset assessment data" for button in app.button)
    next(
        field for field in app.text_input if field.label == "Settings access key"
    ).input("incorrect").run()
    _button(app, "Unlock Settings").click().run()
    assert any("access key is incorrect" in item.value for item in app.error)

    next(
        field for field in app.text_input if field.label == "Settings access key"
    ).input("CMU#AB39").run()
    _button(app, "Unlock Settings").click().run()

    reset_button = _button(app, "Reset assessment data")
    assert not app.exception
    assert reset_button.disabled
    next(
        field for field in app.text_input if field.label == "Type RESET to confirm"
    ).input("RESET").run()
    _button(app, "Reset assessment data").click().run()

    assert any("Assessment data reset complete" in item.value for item in app.success)
    with sqlite3.connect(database_path) as database:
        assert (
            database.execute("SELECT COUNT(*) FROM assessment_runs").fetchone()[0] == 0
        )

    _button(app, "Load dashboard demo data").click().run()
    assert any("Dashboard demo data is ready" in item.value for item in app.success)
    with sqlite3.connect(database_path) as database:
        assert (
            database.execute("SELECT COUNT(*) FROM assessment_runs").fetchone()[0] == 12
        )


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
    # The rejection table is a themed HTML table (ak-table), not a native
    # st.dataframe, so it shows up in app.markdown rather than app.dataframe.
    # (Match the HTML attribute, not just the class name, so the injected
    # theme.css — which also contains the literal string "ak-table-wrap" in
    # its own selectors — doesn't count as a second match.)
    assert sum(
        'class="ak-table-wrap"' in str(item.value) for item in app.markdown
    ) == 1


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
            for table in (
                "features",
                "scores",
                "explanations",
                "decisions",
                "assessment_runs",
                "assessment_decision_links",
            )
        ]
    assert counts == [1, 1, 1, 0, 1, 0]

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
            for table in (
                "features",
                "scores",
                "explanations",
                "decisions",
                "assessment_runs",
                "assessment_decision_links",
            )
        ]
        audit = connection.execute(
            """
            SELECT source_key, processed_count, valid_count, rejected_count
            FROM assessment_runs
            """
        ).fetchone()
    assert counts == [1, 1, 1, 1, 1, 1]
    assert audit[0] == "demo"
    assert audit[1] == audit[2] + audit[3]
    assert app.session_state["decision_saved"] is True
    assert app.session_state["session_history"][0]["decision"] == "Review"

    _button(app, "Start new assessment").click().run()

    assert not app.exception
    assert app.session_state["assessment_step"] == 0
    assert app.session_state["decision_value"] is None
    assert app.session_state["decision_rationale"] == ""
    assert app.session_state["session_history"][0]["decision"] == "Review"
    assert _has_markdown(app, "Applicant")


def test_upload_adapters_reject_empty_input() -> None:
    with pytest.raises(UIInputError, match="empty"):
        read_csv_records(b"")
    with pytest.raises(UIInputError, match="at least one"):
        parse_sms_records("  \n")
