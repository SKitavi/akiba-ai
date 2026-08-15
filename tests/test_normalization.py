"""Tests for canonical transaction normalization and validation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.domain.transactions import (
    FEATURE_TRANSACTION_COLUMNS,
    NormalizedTransaction,
    TransactionProvider,
    TransactionType,
)
from src.features.build_features import build_feature_table
from src.ingestion.normalization import (
    TransactionContext,
    TransactionNormalizationError,
    normalize_transaction,
    normalize_transactions,
)
from src.ingestion.ocr_parser import (
    OCRExtractionError,
    extract_text_from_image,
)
from src.ingestion.sms_parser import parse_sms_message


def _valid_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "applicant_id": "APP_0001",
        "timestamp": "2026-01-02 10:30:00",
        "provider": "M-Pesa",
        "tx_type": "CASH_IN",
        "amount": 2500.0,
        "post_balance": 5000.0,
        "tx_id": "ABC1234567",
        "raw_text": "Synthetic supported record",
    }
    record.update(overrides)
    return record


def test_valid_transaction_normalization() -> None:
    transaction = normalize_transaction(_valid_record())

    assert transaction == NormalizedTransaction(
        applicant_id="APP_0001",
        timestamp=datetime(2026, 1, 2, 10, 30),
        provider=TransactionProvider.MPESA,
        tx_type=TransactionType.CASH_IN,
        amount=2500.0,
        post_balance=5000.0,
        transaction_id="ABC1234567",
        raw_text="Synthetic supported record",
    )


def test_explicit_context_associates_applicant() -> None:
    record = _valid_record()
    record.pop("applicant_id")

    transaction = normalize_transaction(
        record, context=TransactionContext(applicant_id="APP_CONTEXT")
    )

    assert transaction.applicant_id == "APP_CONTEXT"


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("mpesa", TransactionProvider.MPESA),
        ("M-PESA", TransactionProvider.MPESA),
        ("mtn momo", TransactionProvider.MTN_MOMO),
        ("MTN_MoMo", TransactionProvider.MTN_MOMO),
    ],
)
def test_provider_aliases_are_canonical(
    provider: str, expected: TransactionProvider
) -> None:
    assert normalize_transaction(_valid_record(provider=provider)).provider is expected


def test_transaction_type_is_canonicalized_safely() -> None:
    transaction = normalize_transaction(_valid_record(tx_type="p2p-send"))

    assert transaction.tx_type is TransactionType.P2P_SEND


@pytest.mark.parametrize("amount", [None, "bad", -1, 0, float("inf")])
def test_invalid_amount_is_rejected(amount: object) -> None:
    with pytest.raises(TransactionNormalizationError, match="amount"):
        normalize_transaction(_valid_record(amount=amount))


@pytest.mark.parametrize("balance", [None, "bad", -1, float("nan")])
def test_invalid_balance_is_rejected(balance: object) -> None:
    with pytest.raises(TransactionNormalizationError, match="post_balance"):
        normalize_transaction(_valid_record(post_balance=balance))


def test_missing_timestamp_is_rejected() -> None:
    with pytest.raises(TransactionNormalizationError) as exc_info:
        normalize_transaction(_valid_record(timestamp=None))

    assert exc_info.value.code == "missing_timestamp"


def test_malformed_timestamp_is_rejected() -> None:
    with pytest.raises(TransactionNormalizationError) as exc_info:
        normalize_transaction(_valid_record(timestamp="yesterday"))

    assert exc_info.value.code == "invalid_timestamp"


def test_missing_applicant_is_rejected() -> None:
    with pytest.raises(TransactionNormalizationError) as exc_info:
        normalize_transaction(_valid_record(applicant_id=None))

    assert exc_info.value.code == "missing_applicant"


def test_context_applicant_conflict_is_rejected() -> None:
    with pytest.raises(TransactionNormalizationError) as exc_info:
        normalize_transaction(
            _valid_record(),
            context=TransactionContext(applicant_id="APP_OTHER"),
        )

    assert exc_info.value.code == "applicant_mismatch"


def test_unsupported_transaction_type_is_rejected() -> None:
    with pytest.raises(TransactionNormalizationError) as exc_info:
        normalize_transaction(_valid_record(tx_type="MERCHANT"))

    assert exc_info.value.code == "unsupported_transaction_type"


def test_extra_parser_fields_are_ignored() -> None:
    transaction = normalize_transaction(
        _valid_record(counterparty="Merchant", fee=12.0, unknown="ignored")
    )

    assert transaction.amount == 2500.0


def test_batch_collects_rejected_records_and_counts() -> None:
    result = normalize_transactions(
        [_valid_record(), _valid_record(amount=-1), _valid_record(tx_type="UNKNOWN")]
    )

    assert result.processed_count == 3
    assert result.valid_count == 1
    assert result.rejected_count == 2
    assert [rejected.record_index for rejected in result.rejected_transactions] == [
        1,
        2,
    ]


def test_batch_ordering_is_deterministic() -> None:
    later = _valid_record(timestamp="2026-03-01 00:00:00", tx_id="LATER")
    earlier = _valid_record(timestamp="2026-01-01 00:00:00", tx_id="EARLIER")

    result = normalize_transactions([later, earlier])

    assert [item.transaction_id for item in result.valid_transactions] == [
        "EARLIER",
        "LATER",
    ]


def test_dataframe_conversion_matches_feature_schema() -> None:
    result = normalize_transactions(pd.DataFrame([_valid_record()]))
    frame = result.to_dataframe()

    assert set(FEATURE_TRANSACTION_COLUMNS).issubset(frame.columns)
    assert frame.loc[0, "applicant_id"] == "APP_0001"
    assert frame.loc[0, "tx_type"] == "CASH_IN"


def test_supported_sms_to_normalization_to_feature_builder() -> None:
    sms_messages = [
        (
            "AAA11BBB22 Confirmed. You have received Ksh2,500.00 from 0712345678 "
            "on 1/1/26 at 10:00 AM. New M-PESA balance is Ksh5,000.00."
        ),
        (
            "CCC33DDD44 Confirmed. Ksh500.00 sent to 0798765432 "
            "on 2/1/26 at 11:00 AM. New M-PESA balance is Ksh4,500.00. "
            "Transaction cost, Ksh12.00."
        ),
    ]
    parsed = [parse_sms_message(message) for message in sms_messages]

    result = normalize_transactions(
        parsed, context=TransactionContext(applicant_id="APP_SMS")
    )
    features = build_feature_table(result.to_dataframe())

    assert result.rejected_count == 0
    assert result.valid_count == 2
    assert features.loc[0, "applicant_id"] == "APP_SMS"
    assert features.loc[0, "inflow_total"] == 2500.0
    assert features.loc[0, "outflow_total"] == 500.0


def test_real_ocr_path_does_not_silently_return_mock_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("src.ingestion.ocr_parser.HAS_TESSERACT", False)
    monkeypatch.setattr("src.ingestion.ocr_parser.HAS_PILLOW", False)

    with pytest.raises(OCRExtractionError, match="Could not extract"):
        extract_text_from_image(tmp_path / "missing.png")


def test_mock_ocr_fallback_requires_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("src.ingestion.ocr_parser.HAS_TESSERACT", False)
    monkeypatch.setattr("src.ingestion.ocr_parser.HAS_PILLOW", False)

    text = extract_text_from_image(
        tmp_path / "sample_momo.png", allow_mock_fallback=True
    )

    assert "MTN_MOMO TRANSACTION RECORD" in text
