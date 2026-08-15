"""Validated, offline loading of AkibaAI XGBoost model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import xgboost as xgb

from src.features.build_features import FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "xgb_v1.json"


class ModelArtifactError(RuntimeError):
    """Base error for model artifact loading and validation failures."""


class ModelArtifactNotFoundError(FileNotFoundError):
    """Raised when the configured model artifact does not exist."""


class ModelMetadataError(ModelArtifactError):
    """Raised when model sidecar metadata is present but invalid."""


class ModelSchemaError(ModelArtifactError):
    """Raised when an artifact does not use AkibaAI's canonical features."""


@dataclass(frozen=True)
class ModelBundle:
    """Loaded classifier plus validated application-facing model information."""

    model: xgb.XGBClassifier
    model_path: Path
    model_version: str
    metadata: Mapping[str, Any]
    feature_names: tuple[str, ...]
    schema_verified: bool


def resolve_model_path(model_path: Path | str | None = None) -> Path:
    """Resolve model path by explicit argument, environment, then default."""
    if model_path is not None:
        return Path(model_path).expanduser()
    configured_path = os.environ.get("MODEL_PATH")
    if configured_path:
        return Path(configured_path).expanduser()
    return DEFAULT_MODEL_PATH


def _load_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        return {}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelMetadataError(
            f"Could not read valid model metadata from '{metadata_path}'."
        ) from exc
    if not isinstance(metadata, dict):
        raise ModelMetadataError("Model metadata must be a JSON object.")
    return metadata


def _validate_metadata(metadata: Mapping[str, Any]) -> tuple[str, bool]:
    version = metadata.get("model_version", "unknown")
    if not isinstance(version, str) or not version.strip():
        raise ModelMetadataError("model_version must be a non-empty string.")

    metadata_features = metadata.get("feature_columns")
    schema_verified = False
    if metadata_features is not None:
        if not isinstance(metadata_features, list) or not all(
            isinstance(feature, str) for feature in metadata_features
        ):
            raise ModelMetadataError("feature_columns must be a list of strings.")
        if metadata_features != FEATURE_COLUMNS:
            raise ModelSchemaError(
                "Model metadata feature_columns do not match canonical FEATURE_COLUMNS."
            )
        schema_verified = True

    metadata_feature_count = metadata.get("n_features")
    if metadata_feature_count is not None:
        if not isinstance(metadata_feature_count, int):
            raise ModelMetadataError("n_features must be an integer.")
        if metadata_feature_count != len(FEATURE_COLUMNS):
            raise ModelSchemaError(
                "Model metadata n_features does not match the canonical feature count."
            )
        schema_verified = True

    return version.strip(), schema_verified


def load_model_bundle(model_path: Path | str | None = None) -> ModelBundle:
    """Load an XGBoost artifact, sidecar metadata, version, and feature schema.

    Missing metadata is supported for legacy artifacts. Their version is
    explicitly ``unknown`` and schema verification relies on booster feature
    names when available; no version is guessed from the filename.
    """
    resolved_path = resolve_model_path(model_path)
    if not resolved_path.is_file():
        raise ModelArtifactNotFoundError(f"Model artifact not found: {resolved_path}")

    model = xgb.XGBClassifier()
    try:
        model.load_model(str(resolved_path))
    except (OSError, ValueError, xgb.core.XGBoostError) as exc:
        raise ModelArtifactError(
            f"Could not load XGBoost model artifact '{resolved_path}'."
        ) from exc

    metadata_path = resolved_path.with_suffix(".meta.json")
    metadata = _load_metadata(metadata_path)
    model_version, metadata_verified = _validate_metadata(metadata)

    booster_feature_names = model.get_booster().feature_names
    booster_verified = booster_feature_names is not None
    if (
        booster_feature_names is not None
        and list(booster_feature_names) != FEATURE_COLUMNS
    ):
        raise ModelSchemaError(
            "Model artifact feature names do not match canonical FEATURE_COLUMNS."
        )

    return ModelBundle(
        model=model,
        model_path=resolved_path,
        model_version=model_version,
        metadata=MappingProxyType(dict(metadata)),
        feature_names=tuple(FEATURE_COLUMNS),
        schema_verified=metadata_verified or booster_verified,
    )
