"""Validation and normalization between parsed input and canonical transactions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
import re
from typing import Any, Iterable, Mapping

import pandas as pd

from src.domain.transactions import (
    NormalizedTransaction,
    TransactionProvider,
    TransactionType,
    transactions_to_dataframe,
)


@dataclass(frozen=True)
class TransactionContext:
    """Explicit caller-supplied values used only when a record omits them."""

    applicant_id: str | None = None
    timestamp: datetime | str | None = None
    provider: TransactionProvider | str | None = None
    tx_type: TransactionType | str | None = None


@dataclass(frozen=True)
class RejectedTransaction:
    """One rejected input record and its understandable validation reason."""

    record_index: int
    code: str
    message: str


@dataclass(frozen=True)
class NormalizationWarning:
    """Non-fatal information about an accepted input record."""

    record_index: int
    code: str
    message: str


@dataclass(frozen=True)
class NormalizationResult:
    """Batch normalization result suitable for application or UI reporting."""

    valid_transactions: tuple[NormalizedTransaction, ...]
    rejected_transactions: tuple[RejectedTransaction, ...]
    warnings: tuple[NormalizationWarning, ...] = ()

    @property
    def processed_count(self) -> int:
        return len(self.valid_transactions) + len(self.rejected_transactions)

    @property
    def valid_count(self) -> int:
        return len(self.valid_transactions)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_transactions)

    def to_dataframe(self) -> pd.DataFrame:
        """Return valid transactions in the feature builder's expected schema."""
        return transactions_to_dataframe(self.valid_transactions)


class TransactionNormalizationError(ValueError):
    """Raised when one input record cannot be normalized safely."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


_PROVIDER_ALIASES: dict[str, TransactionProvider] = {
    "mpesa": TransactionProvider.MPESA,
    "mtnmomo": TransactionProvider.MTN_MOMO,
    "momo": TransactionProvider.MTN_MOMO,
}

_TIMESTAMP_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%d/%m/%y at %I:%M %p",
    "%d/%m/%y %I:%M %p",
)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _context_value(
    record: Mapping[str, Any],
    field_name: str,
    context_value: object,
) -> object:
    record_value = record.get(field_name)
    if not _is_missing(record_value):
        return record_value
    return context_value


def _normalize_applicant_id(value: object) -> str:
    if _is_missing(value):
        raise TransactionNormalizationError(
            "missing_applicant", "applicant_id is required."
        )
    if not isinstance(value, str):
        raise TransactionNormalizationError(
            "invalid_applicant", "applicant_id must be a non-empty string."
        )
    applicant_id = value.strip()
    if not applicant_id:
        raise TransactionNormalizationError(
            "missing_applicant", "applicant_id is required."
        )
    return applicant_id


def _normalize_provider(value: object) -> TransactionProvider:
    if isinstance(value, TransactionProvider):
        return value
    if _is_missing(value):
        raise TransactionNormalizationError("missing_provider", "provider is required.")
    if not isinstance(value, str):
        raise TransactionNormalizationError(
            "unsupported_provider", f"Unsupported provider: {value!r}."
        )
    key = re.sub(r"[^a-z0-9]", "", value.lower())
    try:
        return _PROVIDER_ALIASES[key]
    except KeyError as exc:
        raise TransactionNormalizationError(
            "unsupported_provider", f"Unsupported provider: {value!r}."
        ) from exc


def _normalize_transaction_type(value: object) -> TransactionType:
    if isinstance(value, TransactionType):
        return value
    if _is_missing(value):
        raise TransactionNormalizationError(
            "missing_transaction_type", "tx_type is required."
        )
    if not isinstance(value, str):
        raise TransactionNormalizationError(
            "unsupported_transaction_type", f"Unsupported tx_type: {value!r}."
        )
    canonical_value = re.sub(r"[\s-]+", "_", value.strip().upper())
    try:
        return TransactionType(canonical_value)
    except ValueError as exc:
        raise TransactionNormalizationError(
            "unsupported_transaction_type", f"Unsupported tx_type: {value!r}."
        ) from exc


def _normalize_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if _is_missing(value):
        raise TransactionNormalizationError(
            "missing_timestamp", "timestamp is required."
        )
    if not isinstance(value, str):
        raise TransactionNormalizationError(
            "invalid_timestamp", f"Invalid timestamp: {value!r}."
        )
    for timestamp_format in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(value.strip(), timestamp_format)
        except ValueError:
            continue
    raise TransactionNormalizationError(
        "invalid_timestamp",
        f"Unsupported timestamp format: {value!r}. Use ISO YYYY-MM-DD HH:MM:SS.",
    )


def _normalize_money(value: object, field_name: str, minimum: float) -> float:
    if _is_missing(value):
        raise TransactionNormalizationError(
            f"missing_{field_name}", f"{field_name} is required."
        )
    if isinstance(value, bool):
        raise TransactionNormalizationError(
            f"invalid_{field_name}", f"{field_name} must be a finite number."
        )
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise TransactionNormalizationError(
            f"invalid_{field_name}", f"{field_name} must be a finite number."
        ) from exc
    if not isfinite(numeric_value) or numeric_value < minimum:
        comparator = "greater than zero" if minimum > 0.0 else "zero or greater"
        raise TransactionNormalizationError(
            f"invalid_{field_name}", f"{field_name} must be finite and {comparator}."
        )
    return numeric_value


def normalize_transaction(
    record: Mapping[str, Any],
    context: TransactionContext | None = None,
) -> NormalizedTransaction:
    """Validate and adapt one parsed or structured record without inventing data."""
    if not isinstance(record, Mapping):
        raise TransactionNormalizationError(
            "invalid_record", "Each transaction record must be a mapping."
        )
    context = context or TransactionContext()

    record_applicant = record.get("applicant_id")
    if (
        not _is_missing(record_applicant)
        and not _is_missing(context.applicant_id)
        and str(record_applicant).strip() != str(context.applicant_id).strip()
    ):
        raise TransactionNormalizationError(
            "applicant_mismatch",
            "Record applicant_id does not match the caller-supplied applicant_id.",
        )

    applicant_id = _normalize_applicant_id(
        _context_value(record, "applicant_id", context.applicant_id)
    )
    timestamp = _normalize_timestamp(
        _context_value(record, "timestamp", context.timestamp)
    )
    provider = _normalize_provider(_context_value(record, "provider", context.provider))
    tx_type = _normalize_transaction_type(
        _context_value(record, "tx_type", context.tx_type)
    )
    amount = _normalize_money(record.get("amount"), "amount", minimum=0.0000001)

    post_balance_value = record.get("post_balance")
    parser_balance_value = record.get("balance")
    if not _is_missing(post_balance_value) and not _is_missing(parser_balance_value):
        normalized_post_balance = _normalize_money(
            post_balance_value, "post_balance", minimum=0.0
        )
        normalized_parser_balance = _normalize_money(
            parser_balance_value, "post_balance", minimum=0.0
        )
        if normalized_post_balance != normalized_parser_balance:
            raise TransactionNormalizationError(
                "balance_conflict",
                "post_balance and parsed balance contain different values.",
            )
    balance_value = (
        parser_balance_value if _is_missing(post_balance_value) else post_balance_value
    )
    post_balance = _normalize_money(balance_value, "post_balance", minimum=0.0)

    transaction_id_value = record.get("transaction_id", record.get("tx_id"))
    transaction_id = (
        None if _is_missing(transaction_id_value) else str(transaction_id_value).strip()
    )
    raw_text_value = record.get("raw_text", record.get("sms_text"))
    raw_text = None if _is_missing(raw_text_value) else str(raw_text_value)

    return NormalizedTransaction(
        applicant_id=applicant_id,
        timestamp=timestamp,
        provider=provider,
        tx_type=tx_type,
        amount=amount,
        post_balance=post_balance,
        transaction_id=transaction_id,
        raw_text=raw_text,
    )


def normalize_transactions(
    records: Iterable[Mapping[str, Any]] | pd.DataFrame,
    context: TransactionContext | None = None,
) -> NormalizationResult:
    """Normalize a batch, collecting rejected records instead of aborting early."""
    input_records: Iterable[Mapping[str, Any]]
    if isinstance(records, pd.DataFrame):
        input_records = records.to_dict(orient="records")
    else:
        input_records = records

    valid_transactions: list[NormalizedTransaction] = []
    rejected_transactions: list[RejectedTransaction] = []
    for record_index, record in enumerate(input_records):
        try:
            valid_transactions.append(normalize_transaction(record, context=context))
        except TransactionNormalizationError as exc:
            rejected_transactions.append(
                RejectedTransaction(
                    record_index=record_index,
                    code=exc.code,
                    message=str(exc),
                )
            )

    valid_transactions.sort(
        key=lambda transaction: (
            transaction.timestamp,
            transaction.applicant_id,
            transaction.transaction_id or "",
        )
    )
    return NormalizationResult(
        valid_transactions=tuple(valid_transactions),
        rejected_transactions=tuple(rejected_transactions),
    )
