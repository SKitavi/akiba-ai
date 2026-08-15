"""Current-session assessment history view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui.components import render_empty_state, render_page_header
from src.ui.state import reset_assessment


def render_history() -> None:
    """Render only records available through the supported session contract."""
    render_page_header(
        "Assessment history",
        "Review assessments created in this browser session. Persisted-history "
        "retrieval is not yet exposed by the frozen backend.",
        eyebrow="Current session",
    )

    history = list(st.session_state.session_history)
    if not history:
        with st.container(border=True):
            render_empty_state(
                "No session history",
                "Saved assessments will appear here after you complete the workflow.",
            )
        if st.button("Start new assessment", type="primary", key="history_start"):
            reset_assessment()
            st.rerun()
        return

    search = st.text_input(
        "Search applicant ID",
        placeholder="APP_0001",
        key="history_search",
    ).strip()
    visible = history
    if search:
        visible = [
            item
            for item in history
            if search.lower() in str(item.get("applicant", "")).lower()
        ]

    if not visible:
        render_empty_state(
            "No matching assessment",
            "Try another applicant ID or clear the search field.",
        )
        return

    frame = pd.DataFrame(visible)[
        ["applicant", "assessed", "model", "risk_score", "decision", "rationale"]
    ]
    st.dataframe(
        frame,
        hide_index=True,
        use_container_width=True,
        column_config={
            "applicant": "Applicant",
            "assessed": "Assessed",
            "model": "Model",
            "risk_score": st.column_config.NumberColumn(
                "Model risk score", format="%.3f"
            ),
            "decision": "Officer decision",
            "rationale": "Rationale",
        },
    )
