"""Persisted operational analytics for the AkibaAI workstation."""

from __future__ import annotations

import sqlite3

import altair as alt
import pandas as pd
import streamlit as st

from src.storage.analytics import AssessmentHistoryItem
from src.ui.components import (
    render_empty_state,
    render_failure_panel,
    render_page_header,
    render_panel_heading,
    render_summary_counters,
    render_table,
)
from src.ui.services import load_assessment_analytics, load_assessment_history
from src.ui.state import Route, navigate, reset_assessment


_SOURCE_LABELS = {
    "demo": "Demonstration data",
    "csv": "CSV upload",
    "sms": "SMS messages",
    "receipt": "Receipt image",
    "unspecified": "Unspecified",
}

_DECISION_LABELS = {
    "APPROVE": "Approved",
    "REVIEW": "Under review",
    "DECLINE": "Declined",
}

_CHART_FONT = 'Inter, "Segoe UI", Arial, sans-serif'
_FOREST = "#0B4A3B"
_GOLD = "#C18B2F"
_INK = "#12110F"
_MUTED = "#77746C"
_GRID = "#E8E5DC"


def _finish_chart(chart: alt.Chart | alt.LayerChart, *, height: int = 220):
    """Apply the shared visual language used by overview analytics."""
    return (
        chart.properties(height=height)
        .configure_view(stroke=None)
        .configure_axis(
            domain=False,
            gridColor=_GRID,
            gridOpacity=1,
            labelColor=_MUTED,
            labelFont=_CHART_FONT,
            labelFontSize=11,
            labelPadding=8,
            tickColor=_GRID,
            tickSize=0,
            titleColor=_MUTED,
            titleFont=_CHART_FONT,
            titleFontSize=11,
            titleFontWeight=500,
            titlePadding=12,
        )
    )


def _decision_activity_chart(frame: pd.DataFrame):
    """Build a compact status chart with labels that remain readable."""
    status_order = frame["Status"].tolist()
    maximum = max(int(frame["Assessments"].max()), 1)
    domain_max = max(maximum * 1.18, maximum + 1)

    bars = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=5, height=24)
        .encode(
            x=alt.X(
                "Assessments:Q",
                title="Assessments",
                scale=alt.Scale(domain=[0, domain_max]),
                axis=alt.Axis(tickMinStep=1),
            ),
            y=alt.Y(
                "Status:N",
                title=None,
                sort=status_order,
                axis=alt.Axis(labelColor=_INK, labelFontWeight=500),
            ),
            color=alt.condition(
                alt.datum.Status == "Waiting",
                alt.value(_GOLD),
                alt.value(_FOREST),
            ),
            tooltip=[
                alt.Tooltip("Status:N", title="Status"),
                alt.Tooltip("Assessments:Q", title="Assessments", format=",d"),
            ],
        )
    )
    labels = (
        alt.Chart(frame)
        .mark_text(
            align="left",
            baseline="middle",
            color=_INK,
            dx=8,
            font=_CHART_FONT,
            fontSize=11,
            fontWeight=700,
        )
        .encode(
            x=alt.X("Assessments:Q"),
            y=alt.Y("Status:N", sort=status_order),
            text=alt.Text("Assessments:Q", format=",d"),
        )
    )
    return _finish_chart(bars + labels)


def _score_distribution_chart(frame: pd.DataFrame):
    """Build an annotated distribution chart with horizontal interval labels."""
    interval_order = frame["Score interval"].tolist()
    maximum = max(int(frame["Assessments"].max()), 1)
    domain_max = max(maximum * 1.18, maximum + 1)

    bars = (
        alt.Chart(frame)
        .mark_bar(
            color=_GOLD,
            cornerRadiusTopLeft=5,
            cornerRadiusTopRight=5,
            size=54,
        )
        .encode(
            x=alt.X(
                "Score interval:N",
                title="Model score interval",
                sort=interval_order,
                axis=alt.Axis(labelAngle=0, labelLimit=100),
            ),
            y=alt.Y(
                "Assessments:Q",
                title="Assessments",
                scale=alt.Scale(domain=[0, domain_max]),
                axis=alt.Axis(tickMinStep=1),
            ),
            tooltip=[
                alt.Tooltip("Score interval:N", title="Score interval"),
                alt.Tooltip("Assessments:Q", title="Assessments", format=",d"),
            ],
        )
    )
    labels = (
        alt.Chart(frame)
        .mark_text(
            baseline="bottom",
            color=_INK,
            dy=-7,
            font=_CHART_FONT,
            fontSize=11,
            fontWeight=700,
        )
        .encode(
            x=alt.X("Score interval:N", sort=interval_order),
            y=alt.Y("Assessments:Q"),
            text=alt.Text("Assessments:Q", format=",d"),
        )
    )
    return _finish_chart(bars + labels)


def _format_timestamp(value: str) -> str:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return value
    return timestamp.strftime("%Y-%m-%d %H:%M UTC")


def _recent_frame(records: tuple[AssessmentHistoryItem, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Applicant": item.applicant_id,
                "Assessed": _format_timestamp(item.assessed_at),
                "Model score": item.risk_score,
                "Officer decision": (
                    _DECISION_LABELS.get(item.decision, item.decision.title())
                    if item.decision
                    else "Waiting for decision"
                ),
            }
            for item in records
        ]
    )


def render_overview() -> None:
    """Render analytics calculated from durable, linked backend records."""
    render_page_header(
        "Loan assessment overview",
        "See saved assessments, pending loan decisions, model scores, and data that needs attention.",
    )

    if st.button("Start new assessment", type="primary", key="overview_start"):
        reset_assessment()
        st.rerun()

    try:
        analytics = load_assessment_analytics()
        recent = load_assessment_history(limit=5)
    except (sqlite3.DatabaseError, OSError, RuntimeError) as exc:
        render_failure_panel(
            "Analytics are temporarily unavailable",
            "The saved assessment data could not be read.",
            technical=str(exc),
        )
        return

    render_summary_counters(
        (
            ("Total assessments", analytics.total_assessments),
            ("Decisions completed", analytics.recorded_decisions),
            ("Waiting for decision", analytics.awaiting_decision),
            ("Data warnings", analytics.warning_count),
        )
    )

    if not analytics.total_assessments:
        render_empty_state(
            "No saved assessments yet",
            "Complete and save an assessment to start building this overview.",
        )
        return

    decisions, scores = st.columns(2, gap="large", vertical_alignment="top")
    with decisions:
        with st.container(border=True):
            render_panel_heading("Loan decision status", "Completed and pending")
            st.markdown(
                '<p class="ak-chart-description">Shows how many loan applications '
                "are approved, declined, under review, or waiting for an officer's "
                "decision.</p>",
                unsafe_allow_html=True,
            )
            decision_frame = pd.DataFrame(
                {
                    "Status": ["Waiting", "Approved", "Under review", "Declined"],
                    "Assessments": [
                        analytics.awaiting_decision,
                        analytics.decision_counts["APPROVE"],
                        analytics.decision_counts["REVIEW"],
                        analytics.decision_counts["DECLINE"],
                    ],
                }
            )
            st.altair_chart(
                _decision_activity_chart(decision_frame),
                use_container_width=True,
                theme=None,
                key="decision_activity_chart",
            )
    with scores:
        with st.container(border=True):
            render_panel_heading("Model score ranges", "Saved assessments")
            st.markdown(
                '<p class="ak-chart-description">Shows how assessments are grouped '
                "by model score. The score supports review but does not make the "
                "final lending decision.</p>",
                unsafe_allow_html=True,
            )
            score_frame = pd.DataFrame(
                {
                    "Score interval": [
                        item.label for item in analytics.score_distribution
                    ],
                    "Assessments": [
                        item.count for item in analytics.score_distribution
                    ],
                }
            )
            st.altair_chart(
                _score_distribution_chart(score_frame),
                use_container_width=True,
                theme=None,
                key="score_distribution_chart",
            )

    sources, quality = st.columns(2, gap="large", vertical_alignment="top")
    with sources:
        with st.container(border=True):
            render_panel_heading("Where the data came from", "Saved assessments")
            source_frame = pd.DataFrame(
                {
                    "Data source": [
                        _SOURCE_LABELS.get(key, key.replace("_", " ").title())
                        for key in analytics.source_counts
                    ],
                    "Assessments": list(analytics.source_counts.values()),
                }
            )
            render_table(source_frame)
    with quality:
        with st.container(border=True):
            render_panel_heading(
                "Data quality checks",
                f"{analytics.assessments_with_audit} assessments checked",
            )
            if analytics.assessments_with_audit:
                quality_frame = pd.DataFrame(
                    {
                        "Check": [
                            "Transactions checked",
                            "Transactions accepted",
                            "Transactions not used",
                            "Data warnings",
                        ],
                        "Transactions": [
                            analytics.processed_records,
                            analytics.valid_records,
                            analytics.rejected_records,
                            analytics.warning_count,
                        ],
                    }
                )
                render_table(quality_frame)
            else:
                st.caption(
                    "Data-quality details are not available for these assessments."
                )

    with st.container(border=True):
        render_panel_heading("Recent assessments", f"Latest {len(recent)}")
        render_table(_recent_frame(recent), formatters={"Model score": "{:.3f}"})
        if st.button("View all assessments", key="overview_history"):
            navigate(Route.HISTORY)
            st.rerun()
