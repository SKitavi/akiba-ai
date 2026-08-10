"""Feature engineering stubs for risk scoring.

Purpose: Provide interfaces for velocity and net-stability feature construction.
Owner: Swafiyah/Sharon (Data + ML Engineers).
Sprint day due: Day 3 (Aug 12) - parsing/features/storage milestone.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


# TODO(Swafiyah/Sharon): Build transaction velocity and income stability features.
def build_feature_table(events_df: "pd.DataFrame") -> "pd.DataFrame":
    """Transform normalized events into a model-ready feature table."""
    raise NotImplementedError("Feature engineering is planned for Day 3.")
