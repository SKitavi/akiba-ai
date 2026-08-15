"""OCR wrapper for synthetic receipt image text extraction.

Purpose: Define a thin pytesseract adapter for offline receipt parsing.
Owner: Swafiyah (Data Engineer).
Sprint day due: Day 3 (Aug 12) - parsing/features/storage milestone.
"""

from pathlib import Path
from src.ingestion.ocr_parser import extract_text_from_image


def extract_text_from_receipt(image_path: Path) -> str:
    """Extract text with an explicitly synthetic fallback for demo fixtures."""
    return extract_text_from_image(image_path, allow_mock_fallback=True)
