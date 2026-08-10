"""Evaluation metric reporting stubs.

Purpose: Define precision/recall/F1/AUC-ROC reporting interfaces for model evaluation.
Owner: Joshua (Explainability Engineer).
Sprint day due: Day 4 (Aug 13) - model training + eval milestone.
"""

from typing import Any, Sequence


# TODO(Joshua): Implement metric calculations + JSON/plot export for sprint reporting.
def generate_classification_report(y_true: Sequence[int], y_score: Sequence[float]) -> dict[str, Any]:
    """Return precision, recall, F1, and AUC-ROC metrics for binary scoring."""
    raise NotImplementedError("Evaluation metrics implementation is planned for Day 4.")
