"""Reusable visual components for the AkibaAI Streamlit interface."""

from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st

from src.model.loader import resolve_model_path
from src.ui.state import ROUTE_LABELS, Route
from src.xai.narratives import NarrativeFactor


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


def render_step_bar(current_step: int) -> None:
    """Render the compact assessment progress indicator."""
    labels = ("Applicant", "Transactions", "Validation", "Assessment", "Decision")
    items = []
    for index, label in enumerate(labels):
        if index < current_step:
            state_class = "complete"
            marker = "✓"
        elif index == current_step:
            state_class = "current"
            marker = str(index + 1)
        else:
            state_class = "upcoming"
            marker = str(index + 1)
        items.append(
            f'<span class="ak-step {state_class}"><b>{marker}</b>{escape(label)}</span>'
        )
    st.markdown(
        '<nav class="ak-step-bar" aria-label="Assessment progress">'
        + '<span class="ak-step-separator">/</span>'.join(items)
        + "</nav>",
        unsafe_allow_html=True,
    )


def render_failure_panel(
    title: str, message: str, technical: str | None = None
) -> None:
    """Render a calm failure state with optional developer detail."""
    st.markdown(
        '<div class="ak-failure-panel">'
        '<div class="ak-overline">Action needed</div>'
        f'<div class="ak-failure-title">{escape(title)}</div>'
        f'<div class="ak-failure-copy">{escape(message)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    if technical:
        with st.expander("View technical details"):
            st.code(technical)


def render_validation_counters(
    processed: int,
    valid: int,
    rejected: int,
    warnings: int,
) -> None:
    """Render validation counts with words as well as restrained colour."""
    values = (
        ("Processed", processed, "neutral"),
        ("Valid", valid, "valid"),
        ("Rejected", rejected, "attention" if rejected else "neutral"),
        ("Warnings", warnings, "attention" if warnings else "neutral"),
    )
    columns = st.columns(4, gap="small")
    for column, (label, value, tone) in zip(columns, values):
        with column:
            st.markdown(
                f'<div class="ak-counter {tone}">'
                f'<span class="ak-counter-label">{label}</span>'
                f"<strong>{value:,}</strong>"
                "</div>",
                unsafe_allow_html=True,
            )


def render_metric_rows(rows: tuple[tuple[str, str], ...]) -> None:
    """Render compact label/value rows for financial behaviour."""
    body = "".join(
        '<div class="ak-metric-row">'
        f"<span>{escape(label)}</span><strong>{escape(value)}</strong>"
        "</div>"
        for label, value in rows
    )
    st.markdown(f'<div class="ak-metric-list">{body}</div>', unsafe_allow_html=True)


def render_score_panel(risk_score: float, model_version: str) -> None:
    """Render a neutral model output without invented policy thresholds."""
    position = max(0.0, min(1.0, risk_score)) * 100
    st.markdown(
        '<section class="ak-score-panel" aria-label="Model risk score">'
        '<div class="ak-overline">Model risk score</div>'
        f'<div class="ak-score-value">{risk_score:.3f}</div>'
        '<div class="ak-score-track" aria-hidden="true">'
        f'<span style="left: {position:.1f}%"></span></div>'
        '<div class="ak-score-ends"><span>0.0</span><span>1.0</span></div>'
        "<p>Higher values indicate greater model-estimated risk. This score "
        "supports human review and does not determine the lending decision.</p>"
        f'<div class="ak-score-model">Model version · {escape(model_version)}</div>'
        "</section>",
        unsafe_allow_html=True,
    )


def render_factor_list(
    title: str,
    factors: tuple[NarrativeFactor, ...],
    *,
    direction_label: str,
) -> None:
    """Render backend-ranked localized factors with accessible direction words."""
    st.markdown(f"#### {title}")
    if not factors:
        st.caption("No material factors were returned in this direction.")
        return
    items = "".join(
        '<div class="ak-factor-row">'
        f"<div><strong>{escape(factor.feature_label)}</strong>"
        f"<p>{escape(factor.text)}</p></div>"
        f"<span>{escape(direction_label)}</span>"
        "</div>"
        for factor in factors
    )
    st.markdown(f'<div class="ak-factor-list">{items}</div>', unsafe_allow_html=True)
    with st.expander(f"Technical details · {title.lower()}"):
        for factor in factors:
            st.code(
                f"{factor.feature_name}: value={factor.feature_value:.4g}, "
                f"SHAP={factor.shap_value:+.4g} ({factor.direction.value})"
            )
