"""Tests for SMS parser scaffolding."""

import pytest

from src.ingestion.sms_parser import parse_sms_message


def test_parse_sms_message_stub_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        parse_sms_message("Sample SMS")
