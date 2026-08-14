"""Parse synthetic SACCO-related SMS transaction messages.

Purpose: Provide regex-based SMS parsing interfaces for ingestion pipeline.
Owner: Swafiyah (Data Engineer).
Sprint day due: Day 3 (Aug 12) - parsing/features/storage milestone.
"""

from typing import Any
from src.ingestion.ocr_parser import parse_transaction_text


def parse_sms_message(message: str) -> dict[str, Any]:
    """Parse one SMS record into normalized fields."""
    return parse_transaction_text(message)
