"""SHAP explainability wrapper stubs.

Purpose: Expose TreeExplainer-ready interfaces for model explanation payloads.
Owner: Joshua (Explainability Engineer).
Sprint day due: Day 5 (Aug 14) - SHAP + narratives + dashboard milestone.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


# TODO(Joshua): Instantiate SHAP explainer and compute per-feature contribution values.
def compute_shap_values(model: Any, features_df: "pd.DataFrame") -> Any:
    """Compute SHAP value outputs for a trained tree-based model."""
    raise NotImplementedError("SHAP integration is planned for Day 5.")
