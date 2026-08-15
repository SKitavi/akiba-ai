"""Persisted assessment history view."""

from __future__ import annotations

import sqlite3

import pandas as pd
import streamlit as st

from src.storage.analytics import AssessmentHistoryItem
from src.ui.components import (
    render_empty_state,
    render_failure_panel,
    render_page_header,
)
from src.ui.services import load_assessment_history
from src.ui.state import reset_assessment


_SOURCE_LABELS = {
    "demo": "Demonstration data",
    "csv": "CSV upload",
    "sms": "SMS messages",
    "receipt": "Receipt image",
}


def _format_timestamp(value: str) -> str:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return value
    return timestamp.strftime("%Y-%m-%d %H:%M UTC")


def _history_frame(records: tuple[AssessmentHistoryItem, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "applicant": item.applicant_id,
                "assessed": _format_timestamp(item.assessed_at),
                "model": item.model_version,
                "risk_score": item.risk_score,
                "source": _SOURCE_LABELS.get(item.source_key or "", "Unspecified"),
                "decision": item.decision.title() if item.decision else "Awaiting",
                "rationale": item.rationale or "",
            }
            for item in records
        ]
    )


def render_history() -> None:
    """Render persisted assessment and linked decision records."""
    render_page_header(
        "Assessment history",
        "Review model assessments and linked officer decisions stored on this "
        "workstation.",
        eyebrow="Persisted local records",
    )

    search = st.text_input(
        "Search applicant ID",
        placeholder="APP_0001",
        key="history_search",
    ).strip()
    try:
        history = load_assessment_history(limit=500, applicant_query=search or None)
    except (sqlite3.DatabaseError, OSError, RuntimeError) as exc:
        render_failure_panel(
            "History is temporarily unavailable",
            "The local assessment database could not be read.",
            technical=str(exc),
        )
        return

    if not history:
        render_empty_state(
            "No matching assessments" if search else "No persisted history",
            (
                "Try another applicant ID or clear the search field."
                if search
                else "Saved assessments will appear here after you complete the workflow."
            ),
        )
        if not search and st.button(
            "Start new assessment", type="primary", key="history_start"
        ):
            reset_assessment()
            st.rerun()
        return

    st.caption(f"Showing {len(history)} persisted assessment records")
    st.dataframe(
        _history_frame(history),
        hide_index=True,
        use_container_width=True,
        column_config={
            "applicant": "Applicant",
            "assessed": "Assessed",
            "model": "Model",
            "risk_score": st.column_config.NumberColumn(
                "Model risk score", format="%.3f"
            ),
            "source": "Evidence source",
            "decision": "Officer decision",
            "rationale": "Rationale",
        },
    )
