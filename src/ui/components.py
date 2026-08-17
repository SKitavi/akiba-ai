"""Reusable visual components for the AkibaAI Streamlit interface."""

from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st

from src.model.loader import resolve_model_path
from src.ui.state import ROUTE_LABELS, Route
from src.xai.narratives import NarrativeFactor


_THEME_PATH = Path(__file__).with_name("theme.css")

_ICON_PATHS: dict[str, str] = {
    "overview": (
        '<rect x="3" y="3" width="7" height="7" rx="1.5"/>'
        '<rect x="14" y="3" width="7" height="7" rx="1.5"/>'
        '<rect x="3" y="14" width="7" height="7" rx="1.5"/>'
        '<rect x="14" y="14" width="7" height="7" rx="1.5"/>'
    ),
    "assessment": (
        '<circle cx="12" cy="12" r="9"/>'
        '<line x1="12" y1="8" x2="12" y2="16"/>'
        '<line x1="8" y1="12" x2="16" y2="12"/>'
    ),
    "history": (
        '<circle cx="12" cy="12" r="9"/>'
        '<line x1="12" y1="7" x2="12" y2="12"/>'
        '<line x1="12" y1="12" x2="16" y2="14"/>'
    ),
    "check": '<polyline points="4 12 9 17 20 6"/>',
    "alert": (
        '<path d="M12 3L22 20L2 20Z"/>'
        '<line x1="12" y1="9" x2="12" y2="14"/>'
        '<line x1="12" y1="17" x2="12" y2="17.01"/>'
    ),
    "layers": (
        '<rect x="4" y="4" width="16" height="4" rx="1"/>'
        '<rect x="4" y="10" width="16" height="4" rx="1"/>'
        '<rect x="4" y="16" width="16" height="4" rx="1"/>'
    ),
    "lock": (
        '<rect x="5" y="11" width="14" height="9" rx="2"/>'
        '<path d="M8 11V7a4 4 0 0 1 8 0v4"/>'
    ),
    "cpu": (
        '<rect x="6" y="6" width="12" height="12" rx="2"/>'
        '<rect x="9" y="9" width="6" height="6" rx="1"/>'
        '<line x1="12" y1="2" x2="12" y2="6"/>'
        '<line x1="12" y1="18" x2="12" y2="22"/>'
        '<line x1="2" y1="12" x2="6" y2="12"/>'
        '<line x1="18" y1="12" x2="22" y2="12"/>'
    ),
}


def _icon(name: str, size: int = 16) -> str:
    """Render a small inline stroke icon from the shared local icon set."""
    return (
        f'<svg class="ak-icon" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{_ICON_PATHS[name]}</svg>'
    )


def inject_theme() -> None:
    """Load the centralized AkibaAI visual system once per app render."""
    css = _THEME_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _model_status() -> str:
    model_path = resolve_model_path()
    return "Model available" if model_path.is_file() else "Model setup required"


def render_sidebar() -> None:
    """Render brand, task navigation, and honest local-environment status."""
    current_route = Route(st.session_state.route)

    with st.sidebar:
        st.markdown(
            '<div class="ak-sb-brand">'
            '<span class="ak-sb-mark">'
            '<svg viewBox="0 0 32 32" aria-hidden="true">'
            '<circle cx="16" cy="16" r="13"/>'
            '<path d="M10 20.5 16 9l6 11.5M12.3 16.3h7.4"/>'
            "</svg></span>"
            '<div class="ak-sb-brand-text">'
            '<span class="ak-sb-name">AkibaAI</span>'
            '<span class="ak-sb-product">Credit intelligence</span>'
            "</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="ak-sb-nav-label">Workspace</div>', unsafe_allow_html=True
        )
        for route, label in ROUTE_LABELS.items():
            if st.button(
                label,
                key=f"nav_{route.value}",
                type="primary" if route is current_route else "secondary",
                use_container_width=True,
            ):
                st.session_state.route = route.value
                st.rerun()

        st.markdown(
            '<div class="ak-sb-status">'
            '<span class="ak-sb-status-label">System status</span>'
            f'<div class="ak-sb-status-row">{_icon("lock", 14)}'
            "<span>Offline workspace</span></div>"
            f'<div class="ak-sb-status-row">{_icon("cpu", 14)}'
            f"<span>{escape(_model_status())}</span></div>"
            '<span class="ak-sb-chip">Synthetic data</span>'
            "</div>",
            unsafe_allow_html=True,
        )


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
        '<div class="ak-page-heading-copy">'
        f"{eyebrow_html}<h1>{escape(title)}</h1>"
        f"<p>{escape(description)}</p></div>"
        '<div class="ak-page-context">'
        '<span class="ak-context-dot"></span>'
        "<span>Local · Synthetic</span></div>"
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
    """Render a connected progress stepper with one active step."""
    labels = ("Applicant", "Transactions", "Validation", "Assessment", "Decision")
    parts: list[str] = []
    for index, label in enumerate(labels):
        if index < current_step:
            state_class = "complete"
            marker = _icon("check", 13)
        elif index == current_step:
            state_class = "current"
            marker = str(index + 1)
        else:
            state_class = "upcoming"
            marker = str(index + 1)
        if index:
            parts.append('<span class="ak-step-connector" aria-hidden="true"></span>')
        parts.append(
            f'<div class="ak-step {state_class}">'
            f'<span class="ak-step-marker">{marker}</span>'
            f'<span class="ak-step-label">{escape(label)}</span></div>'
        )
    st.markdown(
        '<nav class="ak-step-bar" aria-label="Assessment progress">'
        + "".join(parts)
        + "</nav>",
        unsafe_allow_html=True,
    )


def render_failure_panel(
    title: str, message: str, technical: str | None = None
) -> None:
    """Render a calm failure state with optional developer detail."""
    st.markdown(
        '<div class="ak-failure-panel">'
        f'<div class="ak-failure-title">{escape(title)}</div>'
        f'<div class="ak-failure-copy">{escape(message)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    if technical:
        with st.expander("View technical details"):
            st.code(technical)


_SUMMARY_ICON_BY_LABEL: dict[str, str] = {
    "Assessments": "layers",
    "Decisions recorded": "check",
    "Awaiting decision": "history",
    "Ingestion warnings": "alert",
}


def _render_kpi_cards(values: tuple[tuple[str, int, str, str], ...]) -> None:
    """Render labeled totals as bordered, icon-accented KPI cards."""
    cards = "".join(
        f'<div class="ak-kpi-card {tone}">'
        f'<div class="ak-kpi-icon">{_icon(icon_name, 15)}</div>'
        '<div class="ak-kpi-body">'
        f'<span class="ak-kpi-label">{escape(label)}</span>'
        f'<span class="ak-kpi-value">{value:,}</span>'
        "</div></div>"
        for label, value, tone, icon_name in values
    )
    st.markdown(f'<div class="ak-kpi-grid">{cards}</div>', unsafe_allow_html=True)


def render_validation_counters(
    processed: int,
    valid: int,
    rejected: int,
    warnings: int,
) -> None:
    """Render validation counts with words as well as restrained colour."""
    _render_kpi_cards(
        (
            ("Processed", processed, "neutral", "layers"),
            ("Valid", valid, "valid", "check"),
            ("Rejected", rejected, "attention" if rejected else "neutral", "alert"),
            ("Warnings", warnings, "attention" if warnings else "neutral", "alert"),
        )
    )


def render_summary_counters(rows: tuple[tuple[str, int], ...]) -> None:
    """Render four labeled operational totals in the shared summary strip."""
    if len(rows) != 4:
        raise ValueError("A summary strip requires exactly four counters.")
    _render_kpi_cards(
        tuple(
            (label, value, "neutral", _SUMMARY_ICON_BY_LABEL.get(label, "layers"))
            for label, value in rows
        )
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
    """Render a neutral model output without invented policy thresholds.

    The gauge uses a single accent hue with a needle, matching the
    workstation's dial language — but deliberately has no green/amber/red
    zones. There is no approved score-to-risk-band mapping, so nothing here
    implies one.
    """
    position = max(0.0, min(1.0, risk_score)) * 100
    st.markdown(
        '<section class="ak-score-panel" aria-label="Model risk score">'
        '<div class="ak-score-head">'
        '<div><span class="ak-score-kicker">Explainable model output</span>'
        '<div class="ak-score-label">Model risk score</div></div>'
        f'<span class="ak-score-version">Model version: {escape(model_version)}</span>'
        "</div>"
        f'<div class="ak-gauge" style="--pct: {position:.1f}" aria-hidden="true">'
        '<div class="ak-gauge-fill"></div>'
        '<div class="ak-gauge-mask"></div>'
        '<div class="ak-gauge-needle"></div>'
        f'<div class="ak-gauge-value">{risk_score:.3f}</div>'
        "</div>"
        '<div class="ak-gauge-ends"><span>0.0 · Lower</span><span>1.0 · Higher</span></div>'
        "<p>Higher values indicate greater model-estimated risk. This score "
        "supports human review and does not determine the lending decision.</p>"
        '<div class="ak-score-model">Neutral model output · Human review required</div>'
        "</section>",
        unsafe_allow_html=True,
    )


def render_factor_list(
    title: str,
    factors: tuple[NarrativeFactor, ...],
    *,
    direction_label: str,
) -> None:
    """Render backend-ranked localized factors as horizontal SHAP bars."""
    st.markdown(f"#### {title}")
    if not factors:
        st.caption("No material factors were returned in this direction.")
        return
    tone = "danger" if "increas" in direction_label.lower() else "success"
    magnitude = max((abs(factor.shap_value) for factor in factors), default=0.0)
    items = "".join(
        f'<div class="ak-factor-row {tone}">'
        '<div class="ak-factor-row-top">'
        f"<strong>{escape(factor.feature_label)}</strong>"
        f"<span>{factor.shap_value:+.3f}</span>"
        "</div>"
        '<div class="ak-factor-track">'
        '<div class="ak-factor-fill" style="width: '
        f'{(abs(factor.shap_value) / magnitude * 100) if magnitude else 0:.1f}%"></div>'
        "</div>"
        f"<p>{escape(factor.text)}</p>"
        "</div>"
        for factor in factors
    )
    st.markdown(f'<div class="ak-factor-list">{items}</div>', unsafe_allow_html=True)
    with st.expander(f"Technical details — {title.lower()}"):
        for factor in factors:
            st.code(
                f"{factor.feature_name}: value={factor.feature_value:.4g}, "
                f"SHAP={factor.shap_value:+.4g} ({factor.direction.value})"
            )
