"""Generate synthetic SACCO credit-risk records.

Purpose: Define synthetic dataset generation interfaces for offline experiments.
Owner: Swafiyah (Data Engineer).
Sprint day due: Day 2 (Aug 11) - synthetic data milestone.
"""

from pathlib import Path
from typing import Any


# TODO(Swafiyah): Implement Faker/numpy-based synthetic profile + repayment generation.
def generate_synthetic_dataset(output_path: Path, n_rows: int = 1000, seed: int = 42) -> dict[str, Any]:
    """Create and persist synthetic tabular data for pipeline development."""
    raise NotImplementedError("Synthetic data generation is planned for Day 2.")
