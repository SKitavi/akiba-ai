"""Narrative template helpers for explainability outputs.

Purpose: Provide English and Kiswahili narrative builders from feature attributions.
Owner: Joshua (Explainability Engineer).
Sprint day due: Day 5 (Aug 14) - SHAP + narratives + dashboard milestone.
"""

from typing import Mapping


# TODO(Joshua): Implement localized narrative rendering from SHAP/top-feature inputs.
def build_narrative(explanation: Mapping[str, float], language: str = "en") -> str:
    """Generate applicant-friendly explanation text in a supported language."""
    raise NotImplementedError("Narrative generation is planned for Day 5.")
