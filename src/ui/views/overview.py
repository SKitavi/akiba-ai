"""Operational landing view for the AkibaAI workstation."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui.components import (
    render_empty_state,
    render_page_header,
    render_panel_heading,
)
from src.ui.state import Route, navigate, reset_assessment


def render_overview() -> None:
    """Render honest current-session activity without invented analytics."""
    render_page_header(
        "Credit assessment workspace",
        "Review mobile-money behaviour with a local model, explain its output, "
        "and record an independent officer decision.",
        eyebrow="Branch workstation · local processing",
    )

    action, note = st.columns([1, 3], vertical_alignment="center")
    with action:
        if st.button(
            "Start new assessment",
            type="primary",
            use_container_width=True,
            key="overview_start",
        ):
            reset_assessment()
            st.rerun()
    with note:
        st.caption(
            "Demo environment · Synthetic data only · No internet connection required"
        )

    history = list(st.session_state.session_history)
    pending = [item for item in history if not item.get("decision")]
    left, right = st.columns([1, 1], gap="medium")
    with left:
        with st.container(border=True):
            render_panel_heading(
                "Waiting for officer decision", f"{len(pending)} current"
            )
            if not pending:
                render_empty_state(
                    "Nothing waiting for review",
                    "Completed model assessments without a recorded officer decision "
                    "will appear here during this session.",
                )
            else:
                st.dataframe(
                    pd.DataFrame(pending),
                    hide_index=True,
                    use_container_width=True,
                )
    with right:
        with st.container(border=True):
            render_panel_heading("Recent assessments", f"{len(history)} this session")
            if not history:
                render_empty_state(
                    "No assessments yet",
                    "Start a new applicant assessment to create the first local record.",
                )
            else:
                st.dataframe(
                    pd.DataFrame(history[-5:][::-1]),
                    hide_index=True,
                    use_container_width=True,
                )

    with st.container(border=True):
        render_panel_heading("How AkibaAI supports review")
        first, second, third = st.columns(3)
        first.markdown("**1 · Validate evidence**\n\nMalformed records remain visible.")
        second.markdown("**2 · Explain the score**\n\nFactors accompany every result.")
        third.markdown("**3 · Keep judgment human**\n\nNo outcome is preselected.")

    if history and st.button("Open session history", key="overview_history"):
        navigate(Route.HISTORY)
        st.rerun()
