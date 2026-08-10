"""Prediction-time model scoring helpers.

Purpose: Load trained model artifacts and score applicant feature vectors.
Owner: Sharon (ML Engineer).
Sprint day due: Day 4 (Aug 13) - model training + eval milestone.
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


# TODO(Sharon): Load trained model file and compute calibrated risk probabilities.
def score_applicant(model_path: Path, applicant_features: "pd.DataFrame") -> float:
    """Return a single applicant risk score between 0 and 1."""
    raise NotImplementedError("Prediction logic is planned for Day 4.")
