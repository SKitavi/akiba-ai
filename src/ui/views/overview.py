"""Persisted operational analytics for the AkibaAI workstation."""

from __future__ import annotations

import sqlite3

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
                    item.decision.title() if item.decision else "Awaiting"
                ),
            }
            for item in records
        ]
    )


def render_overview() -> None:
    """Render analytics calculated from durable, linked backend records."""
    render_page_header(
        "Credit assessment workspace",
        "Monitor assessment activity and decision follow-through.",
        eyebrow="Local operations overview",
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
            "The local assessment database could not be read.",
            technical=str(exc),
        )
        return

    render_summary_counters(
        (
            ("Assessments", analytics.total_assessments),
            ("Decisions recorded", analytics.recorded_decisions),
            ("Awaiting decision", analytics.awaiting_decision),
            ("Ingestion warnings", analytics.warning_count),
        )
    )

    if not analytics.total_assessments:
        render_empty_state(
            "No persisted assessments yet",
            "Complete and save an assessment to begin building the operational view.",
        )
        return

    decisions, scores = st.columns(2, gap="large", vertical_alignment="top")
    with decisions:
        with st.container(border=True):
            render_panel_heading("Decision activity", "Recorded and outstanding")
            decision_frame = pd.DataFrame(
                {
                    "Status": ["Awaiting", "Approve", "Review", "Decline"],
                    "Assessments": [
                        analytics.awaiting_decision,
                        analytics.decision_counts["APPROVE"],
                        analytics.decision_counts["REVIEW"],
                        analytics.decision_counts["DECLINE"],
                    ],
                }
            )
            st.bar_chart(
                decision_frame,
                x="Status",
                y="Assessments",
                color="#0B4A3B",
                height=270,
            )
    with scores:
        with st.container(border=True):
            render_panel_heading("Model score distribution", "Fixed score intervals")
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
            st.bar_chart(
                score_frame,
                x="Score interval",
                y="Assessments",
                color="#C18B2F",
                height=270,
            )

    sources, quality = st.columns(2, gap="large", vertical_alignment="top")
    with sources:
        with st.container(border=True):
            render_panel_heading("Evidence sources", "Saved assessments")
            source_frame = pd.DataFrame(
                {
                    "Source": [
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
                "Ingestion quality",
                f"{analytics.assessments_with_audit} assessed batches",
            )
            if analytics.assessments_with_audit:
                quality_frame = pd.DataFrame(
                    {
                        "Measure": ["Processed", "Valid", "Rejected", "Warnings"],
                        "Records": [
                            analytics.processed_records,
                            analytics.valid_records,
                            analytics.rejected_records,
                            analytics.warning_count,
                        ],
                    }
                )
                render_table(quality_frame)
            else:
                st.caption("No ingestion-quality metadata is available for these runs.")

    with st.container(border=True):
        render_panel_heading("Recent assessments", f"Latest {len(recent)}")
        render_table(_recent_frame(recent), formatters={"Model score": "{:.3f}"})
        if st.button("Open full history", key="overview_history"):
            navigate(Route.HISTORY)
            st.rerun()
