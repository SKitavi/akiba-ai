"""Reusable visual components for the AkibaAI Streamlit interface."""

from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st

from src.model.loader import resolve_model_path
from src.ui.state import ROUTE_LABELS, Route


_THEME_PATH = Path(__file__).with_name("theme.css")


def inject_theme() -> None:
    """Load the centralized AkibaAI visual system once per app render."""
    css = _THEME_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _model_status() -> str:
    model_path = resolve_model_path()
    return "Model available" if model_path.is_file() else "Model setup required"


def render_app_bar() -> None:
    """Render brand, task navigation, and honest local-environment status."""
    current_route = Route(st.session_state.route)

    brand, navigation, status = st.columns(
        [1.15, 2.7, 2.15], vertical_alignment="center"
    )
    with brand:
        st.markdown(
            '<div class="ak-brand" aria-label="AkibaAI">'
            '<span class="ak-brand-name">AkibaAI</span>'
            '<span class="ak-brand-product">Credit assessment</span>'
            "</div>",
            unsafe_allow_html=True,
        )
    with navigation:
        nav_columns = st.columns(3, gap="small")
        for nav_column, (route, label) in zip(nav_columns, ROUTE_LABELS.items()):
            with nav_column:
                if st.button(
                    label,
                    key=f"nav_{route.value}",
                    type="primary" if route is current_route else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.route = route.value
                    st.rerun()
    with status:
        st.markdown(
            '<div class="ak-status-cluster">'
            '<span class="ak-local-status"><span aria-hidden="true">●</span> Local mode</span>'
            '<span class="ak-demo-chip">Synthetic data</span>'
            f'<span class="ak-model-status">{escape(_model_status())}</span>'
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="ak-app-rule"></div>', unsafe_allow_html=True)


def render_page_header(
    title: str,
    description: str,
    *,
    eyebrow: str | None = None,
) -> None:
    """Render a compact page heading with optional operational context."""
    eyebrow_html = (
        f'<div class="ak-overline">{escape(eyebrow)}</div>' if eyebrow else ""
    )
    st.markdown(
        '<header class="ak-page-header">'
        f"{eyebrow_html}"
        f"<h1>{escape(title)}</h1>"
        f"<p>{escape(description)}</p>"
        "</header>",
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, description: str) -> None:
    """Render a restrained empty state without fake data or illustration."""
    st.markdown(
        '<div class="ak-empty-state">'
        f'<div class="ak-empty-title">{escape(title)}</div>'
        f'<div class="ak-empty-copy">{escape(description)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_panel_heading(title: str, meta: str | None = None) -> None:
    """Render consistent panel heading text inside a bordered container."""
    meta_html = f'<span class="ak-panel-meta">{escape(meta)}</span>' if meta else ""
    st.markdown(
        '<div class="ak-panel-heading">'
        f'<span class="ak-panel-title">{escape(title)}</span>{meta_html}'
        "</div>",
        unsafe_allow_html=True,
    )
