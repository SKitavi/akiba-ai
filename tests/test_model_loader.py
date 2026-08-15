"""Tests for validated XGBoost artifact and metadata loading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.features.build_features import FEATURE_COLUMNS
from src.model.loader import (
    ModelArtifactError,
    ModelArtifactNotFoundError,
    ModelMetadataError,
    ModelSchemaError,
    load_model_bundle,
    resolve_model_path,
)
from src.model.predict import score_applicant
from src.xai.shap_explainer import explain_prediction


@pytest.fixture()
def feature_data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(81)
    features = pd.DataFrame(
        {feature: rng.normal(size=40) for feature in FEATURE_COLUMNS}
    )
    labels = np.array([0, 1] * 20)
    return features, labels


def _save_model(
    model_path: Path,
    features: pd.DataFrame,
    labels: np.ndarray,
    metadata: dict[str, object] | None = None,
) -> None:
    model = xgb.XGBClassifier(n_estimators=6, max_depth=2, n_jobs=1).fit(
        features, labels
    )
    model.save_model(str(model_path))
    if metadata is not None:
        model_path.with_suffix(".meta.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )


def _valid_metadata() -> dict[str, object]:
    return {
        "model_version": "xgb_test_v1",
        "n_features": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
    }


def test_valid_model_and_metadata_load(
    tmp_path: Path, feature_data: tuple[pd.DataFrame, np.ndarray]
) -> None:
    features, labels = feature_data
    model_path = tmp_path / "model.json"
    _save_model(model_path, features, labels, _valid_metadata())

    bundle = load_model_bundle(model_path)

    assert bundle.model_path == model_path
    assert bundle.model_version == "xgb_test_v1"
    assert bundle.feature_names == tuple(FEATURE_COLUMNS)
    assert bundle.schema_verified is True
    assert bundle.metadata["n_features"] == len(FEATURE_COLUMNS)


def test_missing_model_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(ModelArtifactNotFoundError, match="not found"):
        load_model_bundle(tmp_path / "missing.json")


def test_invalid_artifact_fails_clearly(tmp_path: Path) -> None:
    model_path = tmp_path / "invalid.json"
    model_path.write_text("not an xgboost model", encoding="utf-8")

    with pytest.raises(ModelArtifactError, match="Could not load"):
        load_model_bundle(model_path)


def test_metadata_absence_is_supported_without_guessing_version(
    tmp_path: Path, feature_data: tuple[pd.DataFrame, np.ndarray]
) -> None:
    features, labels = feature_data
    model_path = tmp_path / "descriptive_but_untrusted_name.json"
    _save_model(model_path, features, labels)

    bundle = load_model_bundle(model_path)

    assert bundle.model_version == "unknown"
    assert dict(bundle.metadata) == {}
    assert bundle.schema_verified is True


def test_corrupt_metadata_is_rejected(
    tmp_path: Path, feature_data: tuple[pd.DataFrame, np.ndarray]
) -> None:
    features, labels = feature_data
    model_path = tmp_path / "model.json"
    _save_model(model_path, features, labels)
    model_path.with_suffix(".meta.json").write_text("{broken", encoding="utf-8")

    with pytest.raises(ModelMetadataError, match="valid model metadata"):
        load_model_bundle(model_path)


def test_metadata_feature_schema_mismatch_is_rejected(
    tmp_path: Path, feature_data: tuple[pd.DataFrame, np.ndarray]
) -> None:
    features, labels = feature_data
    model_path = tmp_path / "model.json"
    metadata = _valid_metadata()
    metadata["feature_columns"] = [*FEATURE_COLUMNS[:-1], "unexpected"]
    _save_model(model_path, features, labels, metadata)

    with pytest.raises(ModelSchemaError, match="metadata feature_columns"):
        load_model_bundle(model_path)


def test_model_feature_schema_mismatch_is_rejected(
    tmp_path: Path, feature_data: tuple[pd.DataFrame, np.ndarray]
) -> None:
    features, labels = feature_data
    renamed = features.rename(columns={FEATURE_COLUMNS[0]: "unexpected"})
    model_path = tmp_path / "model.json"
    _save_model(model_path, renamed, labels)

    with pytest.raises(ModelSchemaError, match="artifact feature names"):
        load_model_bundle(model_path)


def test_loaded_bundle_is_usable_by_scoring_and_shap(
    tmp_path: Path, feature_data: tuple[pd.DataFrame, np.ndarray]
) -> None:
    features, labels = feature_data
    model_path = tmp_path / "model.json"
    _save_model(model_path, features, labels, _valid_metadata())
    bundle = load_model_bundle(model_path)
    applicant = features.iloc[[0]]

    score = score_applicant(model_path, applicant)
    explanation = explain_prediction(bundle.model, applicant, top_n=3)

    assert 0.0 <= score <= 1.0
    assert explanation.risk_score == pytest.approx(score)


def test_explicit_model_path_precedes_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    environment_path = tmp_path / "environment.json"
    explicit_path = tmp_path / "explicit.json"
    monkeypatch.setenv("MODEL_PATH", str(environment_path))

    assert resolve_model_path(explicit_path) == explicit_path
    assert resolve_model_path() == environment_path
