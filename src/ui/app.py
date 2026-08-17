"""Professional Streamlit entrypoint for the offline AkibaAI workstation."""

from __future__ import annotations

import streamlit as st

from src.ui.components import inject_theme, render_sidebar
from src.ui.state import Route, initialize_state
from src.ui.views.history import render_history
from src.ui.views.overview import render_overview
from src.ui.views.settings import render_settings


def _render_current_route() -> None:
    route = Route(st.session_state.route)
    if route is Route.OVERVIEW:
        render_overview()
    elif route is Route.HISTORY:
        render_history()
    elif route is Route.SETTINGS:
        render_settings()
    else:
        from src.ui.views.new_assessment import render_new_assessment

        render_new_assessment()


def main() -> None:
    """Configure and render the AkibaAI Streamlit application."""
    st.set_page_config(
        page_title="AkibaAI · Credit assessment",
        page_icon="A",
        layout="wide",
        initial_sidebar_state="auto",
    )
    inject_theme()
    initialize_state()
    render_sidebar()
    _render_current_route()


if __name__ == "__main__":
    main()
