"""OCR wrapper for synthetic receipt image text extraction.

Purpose: Define a thin pytesseract adapter for offline receipt parsing.
Owner: Swafiyah (Data Engineer).
Sprint day due: Day 3 (Aug 12) - parsing/features/storage milestone.
"""

from pathlib import Path


# TODO(Swafiyah): Implement pytesseract preprocessing + extraction flow.
def extract_text_from_receipt(image_path: Path) -> str:
    """Extract OCR text from a synthetic receipt image."""
    raise NotImplementedError("OCR implementation is planned for Day 3.")
