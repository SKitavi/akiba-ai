"""Application settings and guarded local data-management actions."""

from __future__ import annotations

import sqlite3

import streamlit as st

from src.storage.data_management import AssessmentDataResetError
from src.ui.components import (
    render_failure_panel,
    render_page_header,
    render_panel_heading,
    render_summary_counters,
)
from src.ui.services import (
    load_assessment_data_counts,
    load_dashboard_demo_data,
    reset_all_assessment_data,
)
from src.ui.state import clear_assessment_workflow


def _render_notice() -> None:
    notice = st.session_state.pop("settings_notice", None)
    if notice:
        st.success(notice)


def render_settings() -> None:
    """Render local data controls with explicit destructive confirmation."""
    if st.session_state.pop("settings_clear_reset_confirmation", False):
        st.session_state.settings_reset_confirmation = ""
    render_page_header(
        "Settings",
        "Manage assessment data stored on this workstation without changing the "
        "model or application configuration.",
        eyebrow="Workspace administration",
    )
    _render_notice()

    try:
        counts = load_assessment_data_counts()
    except (sqlite3.DatabaseError, OSError, RuntimeError) as exc:
        render_failure_panel(
            "Storage information is unavailable",
            "The local assessment database could not be read.",
            technical=str(exc),
        )
        return

    render_summary_counters(
        (
            ("Assessment runs", counts.assessment_runs),
            ("Scores", counts.scores),
            ("Decisions", counts.decisions),
            ("Explanations", counts.explanations),
        )
    )

    demo, reset = st.columns(2, gap="large")
    with demo:
        with st.container(border=True):
            render_panel_heading("Dashboard demo data", "Synthetic records")
            st.write(
                "Load 12 clearly labelled synthetic assessments with balanced "
                "score intervals and sample officer decisions. Existing records "
                "are preserved."
            )
            if st.button("Load dashboard demo data", type="primary"):
                try:
                    result = load_dashboard_demo_data()
                except (
                    sqlite3.DatabaseError,
                    OSError,
                    RuntimeError,
                    ValueError,
                ) as exc:
                    render_failure_panel(
                        "Demo data was not loaded",
                        "The local database could not complete the operation.",
                        technical=str(exc),
                    )
                else:
                    st.session_state.settings_notice = (
                        f"Dashboard demo data is ready: {result.created_count} "
                        "records created."
                    )
                    st.rerun()

    with reset:
        with st.container(border=True):
            render_panel_heading("Reset assessment data", "Danger zone")
            st.warning(
                "This permanently deletes all local features, scores, explanations, "
                "assessment runs, and officer decisions. The model and application "
                "settings are preserved."
            )
            confirmation = st.text_input(
                "Type RESET to confirm",
                key="settings_reset_confirmation",
                autocomplete="off",
            )
            if st.button(
                "Reset assessment data",
                disabled=confirmation != "RESET",
            ):
                try:
                    deleted = reset_all_assessment_data()
                except (
                    AssessmentDataResetError,
                    sqlite3.DatabaseError,
                    OSError,
                    RuntimeError,
                ) as exc:
                    render_failure_panel(
                        "Assessment data was not reset",
                        "The transaction failed, so existing records were preserved.",
                        technical=str(exc),
                    )
                else:
                    clear_assessment_workflow()
                    st.session_state.session_history = []
                    st.session_state.settings_clear_reset_confirmation = True
                    st.session_state.settings_notice = (
                        f"Assessment data reset complete: {deleted.assessment_runs} "
                        "linked assessments removed."
                    )
                    st.rerun()
