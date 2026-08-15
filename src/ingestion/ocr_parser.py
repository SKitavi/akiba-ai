"""SMS Parsing and OCR Extraction for SACCO Credit Scoring.

This module implements Step 3 of the Akiba AI ingestion pipeline:
1. Generating mock thermal receipts for GUI and system testing.
2. OCR text extraction with explicit errors for normal application ingestion.
3. Regex-based parsing for raw SMS messages and structured receipts.

Author: Senior Python Data Engineer
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Union

# Try to import pytesseract and define a safety flag
try:
    import pytesseract

    HAS_TESSERACT = True
except (ImportError, ModuleNotFoundError):
    HAS_TESSERACT = False

# Try to import Pillow for receipt image generation
try:
    from PIL import Image, ImageDraw, ImageFont

    HAS_PILLOW = True
except (ImportError, ModuleNotFoundError):
    HAS_PILLOW = False


def get_font(size: int = 14) -> Any:
    """Helper to load a clean monospaced font on Windows, falling back to default."""
    if not HAS_PILLOW:
        return None

    # Common Windows monospaced font paths
    font_paths = [
        r"C:\Windows\Fonts\consolab.ttf",  # Consolas Bold
        r"C:\Windows\Fonts\consola.ttf",  # Consolas Regular
        r"C:\Windows\Fonts\cour.ttf",  # Courier New
        r"C:\Windows\Fonts\lucon.ttf",  # Lucida Console
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def get_text_width(text: str, draw: Any, font: Any) -> int:
    """Calculates text width in pixels across different Pillow versions."""
    if not HAS_PILLOW:
        return len(text) * 8

    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]
    elif hasattr(draw, "textlength"):
        return int(draw.textlength(text, font=font))
    else:
        # Fallback for old Pillow versions
        return font.getsize(text)[0]


def generate_sample_receipt_image(
    output_path: Union[str, Path],
    provider: str = "M-Pesa",
    tx_id: Union[str, None] = None,
    amount: float = 2500.0,
    fee: float = 0.0,
    balance: float = 15400.0,
    counterparty: Union[str, None] = None,
) -> None:
    """Generates a realistic thermal receipt PNG image for testing purposes.

    Args:
        output_path: Filepath where the receipt image will be saved.
        provider: 'M-Pesa' or 'MTN_MoMo'.
        tx_id: Custom transaction ID.
        amount: Transaction amount.
        fee: Service fee.
        balance: Account balance.
        counterparty: Recipient or merchant name.
    """
    if not HAS_PILLOW:
        raise ImportError("Pillow (PIL) is required to generate receipt images.")

    # Apply defaults if none provided
    if not tx_id:
        tx_id = "UH13Q2B7N6" if provider == "M-Pesa" else "31196215166"
    if not counterparty:
        counterparty = (
            "Naivas Supermarket" if provider == "M-Pesa" else "NYARUGENGE MARKET"
        )

    # Define receipt fields structured like a thermal POS receipt
    lines_to_draw = [
        ("CENTER", "AKIBA SACCO AGENT"),
        ("CENTER", f"{provider.upper()} TRANSACTION RECORD"),
        ("LINE", "-"),
        ("KEYVAL", "Transaction ID:", tx_id),
        ("KEYVAL", "Date:", "2026-08-13 12:30:15"),
        ("KEYVAL", "Counterparty:", counterparty),
        ("LINE", "-"),
        (
            "KEYVAL",
            "Amount Paid:",
            f"Ksh {amount:,.2f}" if provider == "M-Pesa" else f"{int(amount)} RWF",
        ),
        (
            "KEYVAL",
            "Service Fee:",
            f"Ksh {fee:,.2f}" if provider == "M-Pesa" else f"{int(fee)} RWF",
        ),
        (
            "KEYVAL",
            "Wallet Balance:",
            f"Ksh {balance:,.2f}" if provider == "M-Pesa" else f"{int(balance)} RWF",
        ),
        ("LINE", "-"),
        ("KEYVAL", "Status:", "SUCCESSFUL"),
        ("LINE", "-"),
        ("CENTER", "Offline Verification Receipt"),
        ("CENTER", "Thank You!"),
    ]

    line_height = 25
    margin_top_bottom = 30
    img_width = 380
    img_height = len(lines_to_draw) * line_height + margin_top_bottom * 2

    # Create thermal paper image with off-white color (250, 249, 246)
    img = Image.new("RGB", (img_width, img_height), color=(250, 249, 246))
    draw = ImageDraw.Draw(img)

    font_regular = get_font(size=13)
    font_bold = get_font(size=14)

    y = margin_top_bottom
    for item_type, *args in lines_to_draw:
        if item_type == "CENTER":
            text = args[0]
            w = get_text_width(text, draw, font_bold)
            x = (img_width - w) // 2
            draw.text((x, y), text, fill=(30, 30, 30), font=font_bold)
        elif item_type == "LINE":
            # Draw gray divider line
            draw.line(
                [(20, y + 10), (img_width - 20, y + 10)], fill=(180, 180, 180), width=1
            )
        elif item_type == "KEYVAL":
            key, val = args[0], args[1]
            draw.text((20, y), key, fill=(70, 70, 70), font=font_regular)
            val_w = get_text_width(val, draw, font_bold)
            draw.text(
                (img_width - 20 - val_w, y), val, fill=(30, 30, 30), font=font_bold
            )
        y += line_height

    # Save to disk
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def get_mock_receipt_text(provider: str = "M-Pesa") -> str:
    """Return synthetic receipt text for explicitly requested tests and demos."""
    if provider == "MTN_MoMo":
        return """
AKIBA SACCO AGENT
MTN_MOMO TRANSACTION RECORD
--------------------------
Transaction ID: 31196215166
Date: 2026-08-13 12:30:15
Counterparty: NYARUGENGE MARKET
--------------------------
Amount Paid: 2500 RWF
Service Fee: 0 RWF
Wallet Balance: 14200 RWF
--------------------------
Status: SUCCESSFUL
--------------------------
Offline Verification Receipt
Thank You!
"""
    else:  # Default to M-Pesa
        return """
AKIBA SACCO AGENT
M-PESA TRANSACTION RECORD
--------------------------
Transaction ID: UH13Q2B7N6
Date: 2026-08-13 12:30:15
Counterparty: Naivas Supermarket
--------------------------
Amount Paid: Ksh 2,500.00
Service Fee: Ksh 0.00
Wallet Balance: Ksh 15,400.00
--------------------------
Status: SUCCESSFUL
--------------------------
Offline Verification Receipt
Thank You!
"""


class OCRExtractionError(RuntimeError):
    """Raised when receipt text cannot be extracted without synthetic fallback."""


def extract_text_from_image(
    image_path: Union[str, Path], allow_mock_fallback: bool = False
) -> str:
    """Extract OCR text without silently inventing transaction contents.

    The explicitly synthetic ``ocr_stub`` pathway may set ``allow_mock_fallback``
    for demonstrations. Normal application ingestion raises ``OCRExtractionError``
    if Tesseract is unavailable or extraction fails.

    Args:
        image_path: Path to the receipt image file.
        allow_mock_fallback: Permit synthetic receipt text for an explicit test
                             or demo pathway. Keep disabled for real ingestion.

    Returns:
        Extracted OCR text, or synthetic mock text only when explicitly allowed.

    Raises:
        OCRExtractionError: If extraction fails and mock fallback is disabled.
    """
    path_str = str(image_path).lower()

    # Infer a provider only for the explicitly enabled synthetic fallback.
    provider = "M-Pesa"
    if "momo" in path_str or "mtn" in path_str:
        provider = "MTN_MoMo"

    if HAS_TESSERACT and HAS_PILLOW:
        try:
            with Image.open(image_path) as img:
                extracted = pytesseract.image_to_string(img)
            if isinstance(extracted, str) and extracted.strip():
                return extracted
        except Exception:
            # Convert OCR/library failures into the stable error contract below.
            pass

    if allow_mock_fallback:
        return get_mock_receipt_text(provider)
    raise OCRExtractionError(
        f"Could not extract receipt text from '{image_path}'. "
        "Install/configure Tesseract or use the explicit synthetic OCR stub."
    )


def clean_amount(val: str) -> float:
    """Removes commas and formats string numeric amounts to float."""
    if not val:
        return 0.0
    cleaned = val.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


# Regex patterns to isolate counterparty names
COUNTERPARTY_PATTERNS = [
    # M-Pesa SMS
    r"sent to\s+(.*?)\s+on\s+\d",
    r"received Ksh\s*[\d,.]+\s+from\s+(.*?)\s+on\s+\d",
    r"withdrawn from Agent\s+(.*?)\s+on\s+\d",
    r"Received Ksh\s*[\d,.]+\s+from Agent\s+(.*?)\s+on\s+\d",
    r"paid to\s+(.*?)\s+on\s+\d",
    # MTN MoMo SMS
    r"transferred to\s+(.*?)\s+at\s+\d",
    r"received\s+[\d,]+\s*RWF\s+from\s+(.*?)\s+at\s+\d",
    r"from Agent\s+(.*?)\s+at\s+\d",
    r"from Agent\s+(.*?)\s+\.",
    r"Payment of\s+[\d,]+\s*RWF\s+to\s+(.*?)\s+successful",
    r"Payment of\s+[\d,]+\s*RWF\s+to\s+(.*?)\s+(?:\(.*?\))?\s*successful",
    r"Payment of\s+[\d,]+\s*RWF\s+to\s+(.*?)\s+for\s+.*?\s+successful",
    # Receipts (both M-Pesa & MoMo formats)
    r"(?:Merchant|Counterparty|Recipient|Agent|Paid\s+to|Sent\s+to)\s*:\s*(.*)",
]


def _extract_timestamp(text: str) -> str | None:
    """Extract supported provider or receipt timestamps as canonical ISO text."""
    patterns_and_formats = (
        (
            r"\bDate\s*:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})",
            "%Y-%m-%d %H:%M:%S",
        ),
        (
            r"\bat\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})",
            "%Y-%m-%d %H:%M:%S",
        ),
        (
            r"\bon\s+(\d{1,2}/\d{1,2}/\d{2}\s+at\s+\d{1,2}:\d{2}\s+[AP]M)",
            "%d/%m/%y at %I:%M %p",
        ),
    )
    for pattern, timestamp_format in patterns_and_formats:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                timestamp = datetime.strptime(match.group(1), timestamp_format)
            except ValueError:
                continue
            return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    return None


def _infer_transaction_type(text: str, provider: str | None) -> str | None:
    """Infer only transaction types explicitly identified by supported wording."""
    lower_text = text.lower()
    if "airtime" in lower_text:
        return "AIRTIME"
    if "cash in" in lower_text or re.search(
        r"received\s+ksh[\d,.]+\s+from\s+agent", lower_text
    ):
        return "CASH_IN"
    if "cash out" in lower_text or "withdrawn from agent" in lower_text:
        return "CASH_OUT"
    if (
        provider == "M-Pesa"
        and " for account " in lower_text
        and " sent to " in lower_text
    ):
        return "PAYBILL"
    if provider == "MTN_MoMo" and re.search(
        r"payment of .*\sfor\s.*successful", lower_text
    ):
        return "UTILITY"
    if (
        provider == "MTN_MoMo"
        and "payment of" in lower_text
        and "successful" in lower_text
    ):
        return "MOMOPAY_MERCHANT"
    if provider == "M-Pesa" and " paid to " in lower_text:
        return "BUY_GOODS_TILL"
    if "have received" in lower_text:
        return "P2P_RECEIVE"
    if " sent to " in lower_text or "transferred to" in lower_text:
        return "P2P_SEND"
    return None


def parse_transaction_text(text: str) -> dict[str, Any]:
    """Robust regex engine to parse transactional fields from SMS or receipt OCR text.

    Args:
        text: Raw SMS message string or OCR-extracted receipt text.

    Returns:
        A dictionary containing extracted identifiers, money fields, provider,
        timestamp, transaction type, counterparty, and original text. Fields that
        cannot be extracted safely are returned as ``None``.
    """
    # Normalize duplicate whitespaces but keep lines intact
    normalized_lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned_text = " ".join(normalized_lines)
    lower_text = cleaned_text.lower()

    # 1. Detect Provider
    provider = None
    if any(k in lower_text for k in ["m-pesa", "mpesa", "ksh"]):
        provider = "M-Pesa"
    elif any(k in lower_text for k in ["mtn", "momo", "rwf"]):
        provider = "MTN_MoMo"

    # 2. Extract Transaction ID (tx_id)
    tx_id = None

    # Attempt 1: Explicit Receipt Labels
    tx_match = re.search(
        r"\b(?:Transaction\s*ID|TX\s*ID|TxID|Transaction\s*Ref)\s*[:\-]?\s*([A-Za-z0-9]+)\b",
        cleaned_text,
        re.IGNORECASE,
    )
    if tx_match:
        tx_id = tx_match.group(1).strip()

    # Attempt 2: 10-char Alphanumeric Match (M-Pesa & general random IDs)
    if not tx_id:
        # Match word boundaries for exactly 10 characters
        candidates = re.findall(r"\b([A-Za-z0-9]{10})\b", cleaned_text)
        for cand in candidates:
            cand_upper = cand.upper()
            if cand_upper not in ("SUCCESSFUL", "CONFIRMED", "TRANSFERRED"):
                # Require at least one letter and one number to filter out pure words or numbers
                if any(c.isdigit() for c in cand_upper) and any(
                    c.isalpha() for c in cand_upper
                ):
                    tx_id = cand
                    break

    # Attempt 3: Numeric only match of length 10-12 (MTN MoMo style)
    if not tx_id:
        if provider == "MTN_MoMo" and any(
            k in lower_text
            for k in ("txid", "transaction id", "transaction ref", "reference")
        ):
            num_candidates = re.findall(r"\b(\d{10,12})\b", cleaned_text)
            if num_candidates:
                tx_id = num_candidates[0]

    # 3. Extract Amount, Fee, and Balance
    amount = None
    fee = None
    balance = None

    # Check key-value receipt formats first
    amt_match = re.search(
        r"\b(?:Amount\s*Paid|Amount|Amt)\s*[:\-,]?\s*(?:Ksh|KSH)?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)\b",
        cleaned_text,
        re.IGNORECASE,
    )
    if amt_match:
        amount = clean_amount(amt_match.group(1))

    fee_match = re.search(
        r"\b(?:Service\s*Fee|Fee|Transaction\s*cost)\s*[:\-,]?\s*(?:Ksh|KSH)?\s*"
        r"(\d+(?:,\d{3})*(?:\.\d{1,2})?)\b",
        cleaned_text,
        re.IGNORECASE,
    )
    if fee_match:
        fee = clean_amount(fee_match.group(1))

    bal_match = re.search(
        r"\b(?:Wallet\s*Balance|Remaining\s*Bal|Balance|Bal)\s*[:\-,]?\s*(?:Ksh|KSH)?\s*"
        r"(\d+(?:,\d{3})*(?:\.\d{1,2})?)\b",
        cleaned_text,
        re.IGNORECASE,
    )
    if bal_match:
        balance = clean_amount(bal_match.group(1))

    # Pattern-based Fallbacks for SMS logs (which lack key labels)
    if provider == "M-Pesa":
        # Extract all currency format amounts (e.g., Ksh2,500.00)
        ksh_amounts = re.findall(
            r"(?:Ksh|KSH)\s*([\d,]+\.\d{2})\b", cleaned_text, re.IGNORECASE
        )
        if ksh_amounts:
            if amount is None:
                amount = clean_amount(ksh_amounts[0])
            if balance is None and len(ksh_amounts) >= 2:
                if "balance" in lower_text:
                    balance = clean_amount(ksh_amounts[1])
            if fee is None and len(ksh_amounts) >= 3:
                if "cost" in lower_text or "fee" in lower_text:
                    fee = clean_amount(ksh_amounts[2])

    elif provider == "MTN_MoMo":
        # Extract all RWF amounts
        rwf_matches = re.findall(
            r"\b([\d,]+)\s*RWF\b|\bRWF\s*([\d,]+)\b",
            cleaned_text,
            re.IGNORECASE,
        )
        rwf_amounts = [m[0] or m[1] for m in rwf_matches if m[0] or m[1]]
        if rwf_amounts:
            if amount is None:
                amount = clean_amount(rwf_amounts[0])
            # Parse Fee and Balance using contextual labels if not yet set
            if fee is None:
                momo_fee = re.search(
                    r"Fee\s*:\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)\s*RWF",
                    cleaned_text,
                    re.IGNORECASE,
                )
                if momo_fee:
                    fee = clean_amount(momo_fee.group(1))
            if balance is None:
                momo_bal = re.search(
                    r"Balance\s*:\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)\s*RWF",
                    cleaned_text,
                    re.IGNORECASE,
                )
                if momo_bal:
                    balance = clean_amount(momo_bal.group(1))

    # Default fee to 0.0 if not found
    if fee is None:
        fee = 0.0

    # 4. Extract Counterparty
    counterparty = None
    for pattern in COUNTERPARTY_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            counterparty = match.group(1).strip()
            break

    # Clean up counterparty noise (dates, status terms)
    if counterparty:
        counterparty = re.sub(
            r"\s+successful.*$", "", counterparty, flags=re.IGNORECASE
        )
        counterparty = re.sub(
            r"\s+at\s+\d{4}-\d{2}.*$", "", counterparty, flags=re.IGNORECASE
        )
        counterparty = re.sub(
            r"\s+on\s+\d{1,2}/\d{1,2}/.*$", "", counterparty, flags=re.IGNORECASE
        )
        counterparty = counterparty.strip(". ")

    # Fallback to Airtime or Unknown
    if not counterparty or counterparty.lower() == "self":
        if "airtime" in lower_text:
            counterparty = "Airtime"
        else:
            counterparty = "Unknown"

    return {
        "tx_id": tx_id,
        "amount": amount,
        "fee": fee,
        "balance": balance,
        "counterparty": counterparty,
        "provider": provider,
        "timestamp": _extract_timestamp(cleaned_text),
        "tx_type": _infer_transaction_type(cleaned_text, provider),
        "raw_text": text,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("      AKIBA AI: STEP 3 PIPELINE DEMONSTRATION")
    print("=" * 60)

    # 1. Generate sample thermal receipt images
    mpesa_img = Path("data/raw/sample_receipt_mpesa.png")
    momo_img = Path("data/raw/sample_receipt_momo.png")

    print(f"Generating realistic M-Pesa receipt image -> {mpesa_img}")
    generate_sample_receipt_image(
        output_path=mpesa_img,
        provider="M-Pesa",
        tx_id="UH13Q2B7N6",
        amount=3750.50,
        fee=12.00,
        balance=12400.00,
        counterparty="Mama Mboga Grocery",
    )

    print(f"Generating realistic MTN MoMo receipt image -> {momo_img}")
    generate_sample_receipt_image(
        output_path=momo_img,
        provider="MTN_MoMo",
        tx_id="31196215166",
        amount=15000.00,
        fee=20.00,
        balance=74000.00,
        counterparty="NYARUGENGE MARKET",
    )

    # 2. This explicit demo permits synthetic text if real OCR is unavailable.
    print("\nExtracting OCR text (explicit demo fallback enabled)...")
    mpesa_extracted_text = extract_text_from_image(mpesa_img, allow_mock_fallback=True)
    momo_extracted_text = extract_text_from_image(momo_img, allow_mock_fallback=True)

    print("\n--- Extracted M-Pesa Text (OCR or explicit demo fallback): ---")
    print(mpesa_extracted_text.strip())

    print("\n--- Extracted MTN MoMo Text (OCR or explicit demo fallback): ---")
    print(momo_extracted_text.strip())

    # 3. Parse receipt texts into structured outputs
    print("\nParsing extracted receipt texts into structured dictionaries...")
    mpesa_parsed = parse_transaction_text(mpesa_extracted_text)
    momo_parsed = parse_transaction_text(momo_extracted_text)

    print("\nM-Pesa Structured Receipt Data:")
    for k, v in mpesa_parsed.items():
        if k != "raw_text":
            print(f"  {k:15}: {v}")

    print("\nMTN MoMo Structured Receipt Data:")
    for k, v in momo_parsed.items():
        if k != "raw_text":
            print(f"  {k:15}: {v}")

    # 4. Demonstrate parsing of raw provider SMS format strings
    print("\n" + "-" * 60)
    print("Demonstrating parsing of raw SMS templates...")
    sample_sms_logs = [
        # M-Pesa P2P Send
        "UH13Q2B7N6 Confirmed. Ksh750.00 sent to HARUN MWANGI 0112259522 on 1/8/26 at 5:19 PM. "
        "New M-PESA balance is Ksh0.00. Transaction cost, Ksh12.00.",
        # M-Pesa P2P Receive
        "UH13Q2B7N6 Confirmed. You have received Ksh2,500.00 from JOHN DOE 0712345678 on 2/8/26 "
        "at 1:15 PM. New M-PESA balance is Ksh15,400.00.",
        # MTN MoMo Cash Out
        "TxID:31196215166 Cash Out of 5000 RWF from Agent Agent_888 at 2026-03-14 14:22:10 ."
        "Fee: 20RWF.Balance: 1500RWF.",
        # MTN MoMo P2P Send
        "*165*S*2500 RWF transferred to 25078123456 at 2026-03-14 14:22:10 .Fee: 10RWF.Balance: 12000RWF.",
    ]

    for sms in sample_sms_logs:
        print(f"\nSMS Message: {sms}")
        parsed_sms = parse_transaction_text(sms)
        for k, v in parsed_sms.items():
            if k != "raw_text":
                print(f"  {k:15}: {v}")

    print("=" * 60)
