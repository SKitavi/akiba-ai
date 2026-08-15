"""New-assessment workflow placeholder filled in subsequent UI phases."""

from __future__ import annotations

import streamlit as st

from src.ui.components import render_page_header


def render_new_assessment() -> None:
    """Render the assessment workflow shell while intake is being configured."""
    render_page_header(
        "New assessment",
        "Select an applicant, validate transaction evidence, and run an explained "
        "local assessment.",
        eyebrow="Step 1 of 5 · Applicant",
    )
    with st.container(border=True):
        st.markdown("**Applicant intake**")
        st.caption("The applicant and transaction workflow is ready for configuration.")
