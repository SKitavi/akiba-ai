"""Tests for feature engineering scaffolding."""

import pytest

from src.features.build_features import build_feature_table


def test_build_feature_table_stub_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        build_feature_table(None)  # type: ignore[arg-type]
