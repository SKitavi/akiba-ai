"""Tests for model scaffolding."""

from pathlib import Path

import pytest

from src.model.predict import score_applicant
from src.model.train import train_model


def test_train_model_stub_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        train_model(None, Path("/tmp/model.json"))  # type: ignore[arg-type]


def test_score_applicant_stub_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        score_applicant(Path("/tmp/model.json"), None)  # type: ignore[arg-type]
