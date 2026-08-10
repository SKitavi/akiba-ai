"""Model training entrypoint for the AkibaAI MVP.

Purpose: Define interfaces for fitting baseline credit-risk models (XGBoost target).
Owner: Sharon (ML Engineer).
Sprint day due: Day 4 (Aug 13) - model training + eval milestone.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


# TODO(Sharon): Implement train/validation split, fit, and artifact persistence.
def train_model(features_df: "pd.DataFrame", model_output_path: Path) -> dict[str, Any]:
    """Train an MVP model and persist serialized artifact(s)."""
    raise NotImplementedError("Model training implementation is planned for Day 4.")
