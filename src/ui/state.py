"""Central Streamlit session-state contract for the AkibaAI UI."""

from __future__ import annotations

from enum import Enum
from typing import Final

import streamlit as st


class Route(str, Enum):
    """Task-level application destinations."""

    OVERVIEW = "overview"
    ASSESSMENT = "assessment"
    HISTORY = "history"


ROUTE_LABELS: Final[dict[Route, str]] = {
    Route.OVERVIEW: "Overview",
    Route.ASSESSMENT: "New Assessment",
    Route.HISTORY: "History",
}

ASSESSMENT_STATE_KEYS: Final[tuple[str, ...]] = (
    "assessment_step",
    "applicant_id",
    "applicants_df",
    "source_key",
    "source_records",
    "normalization_result",
    "feature_preview",
    "assessment_result",
    "narrative_language",
    "assessment_saved",
    "persisted_assessment",
    "decision_value",
    "decision_rationale",
    "decision_saved",
    "decision_id",
    "last_error",
)

_DEFAULTS: Final[dict[str, object]] = {
    "route": Route.OVERVIEW.value,
    "assessment_step": 0,
    "applicant_id": None,
    "applicants_df": None,
    "source_key": "demo",
    "source_records": None,
    "normalization_result": None,
    "feature_preview": None,
    "assessment_result": None,
    "narrative_language": "en",
    "assessment_saved": False,
    "persisted_assessment": None,
    "decision_value": None,
    "decision_rationale": "",
    "decision_saved": False,
    "decision_id": None,
    "last_error": None,
    "session_history": [],
}


def initialize_state() -> None:
    """Populate every documented state key exactly once per browser session."""
    for key, value in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, list) else value


def navigate(route: Route) -> None:
    """Move to a task-level route."""
    st.session_state.route = route.value


def reset_assessment() -> None:
    """Clear workflow state without discarding current-session history."""
    for key in ASSESSMENT_STATE_KEYS:
        default = _DEFAULTS[key]
        st.session_state[key] = default.copy() if isinstance(default, list) else default
    navigate(Route.ASSESSMENT)
