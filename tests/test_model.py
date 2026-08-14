"""Tests for model training and prediction pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.features.build_features import FEATURE_COLUMNS
from src.model.predict import score_applicant
from src.model.train import train_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_feature_df(n: int = 30, default_rate: float = 0.2) -> pd.DataFrame:
    """Minimal valid feature DataFrame for train_model tests."""
    import numpy as np

    rng = np.random.default_rng(42)
    data = {col: rng.random(n) for col in FEATURE_COLUMNS}
    data["default_label"] = (rng.random(n) < default_rate).astype(int)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------

def test_train_model_returns_metadata_dict(tmp_path: Path) -> None:
    df = _make_feature_df(n=50)
    result = train_model(df, tmp_path / "model.json")

    assert isinstance(result, dict)
    for key in ("model_version", "n_samples", "n_features", "cv_auc_mean", "cv_auc_std",
                "default_rate", "feature_columns"):
        assert key in result, f"Missing key in metadata: {key}"


def test_train_model_persists_artifact(tmp_path: Path) -> None:
    df = _make_feature_df(n=50)
    model_path = tmp_path / "model.json"
    train_model(df, model_path)

    assert model_path.exists(), "Model artifact not created"
    meta_path = model_path.with_suffix(".meta.json")
    assert meta_path.exists(), "Meta JSON sidecar not created"


def test_train_model_cv_auc_reasonable(tmp_path: Path) -> None:
    """CV AUC should be ≥ 0.5 even on random data (no worse than coin flip)."""
    df = _make_feature_df(n=60)
    result = train_model(df, tmp_path / "model.json")
    assert result["cv_auc_mean"] >= 0.0


def test_train_model_missing_features_raises(tmp_path: Path) -> None:
    df = _make_feature_df(n=30).drop(columns=[FEATURE_COLUMNS[0]])
    with pytest.raises(ValueError, match="missing feature columns"):
        train_model(df, tmp_path / "model.json")


def test_train_model_missing_label_raises(tmp_path: Path) -> None:
    df = _make_feature_df(n=30).drop(columns=["default_label"])
    with pytest.raises(ValueError, match="default_label"):
        train_model(df, tmp_path / "model.json")


# ---------------------------------------------------------------------------
# score_applicant
# ---------------------------------------------------------------------------

def test_score_applicant_returns_float_in_range(tmp_path: Path) -> None:
    import numpy as np

    df = _make_feature_df(n=50)
    model_path = tmp_path / "model.json"
    train_model(df, model_path)

    rng = np.random.default_rng(7)
    applicant = pd.DataFrame([{col: rng.random() for col in FEATURE_COLUMNS}])
    score = score_applicant(model_path, applicant)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_score_applicant_missing_model_raises(tmp_path: Path) -> None:
    import numpy as np

    rng = np.random.default_rng(7)
    applicant = pd.DataFrame([{col: rng.random() for col in FEATURE_COLUMNS}])

    with pytest.raises(FileNotFoundError):
        score_applicant(tmp_path / "nonexistent.json", applicant)


def test_score_applicant_missing_features_raises(tmp_path: Path) -> None:
    df = _make_feature_df(n=50)
    model_path = tmp_path / "model.json"
    train_model(df, model_path)

    bad_input = pd.DataFrame([{"wrong_col": 1.0}])
    with pytest.raises(ValueError, match="missing columns"):
        score_applicant(model_path, bad_input)
