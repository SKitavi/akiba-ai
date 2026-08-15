"""Canonical transaction types and records for the AkibaAI backend."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Iterable

import pandas as pd


class TransactionProvider(str, Enum):
    """Mobile-money providers supported by the MVP."""

    MPESA = "M-Pesa"
    MTN_MOMO = "MTN_MoMo"


class TransactionType(str, Enum):
    """Transaction categories understood by feature engineering."""

    CASH_IN = "CASH_IN"
    P2P_RECEIVE = "P2P_RECEIVE"
    CASH_OUT = "CASH_OUT"
    P2P_SEND = "P2P_SEND"
    BUY_GOODS_TILL = "BUY_GOODS_TILL"
    MOMOPAY_MERCHANT = "MOMOPAY_MERCHANT"
    PAYBILL = "PAYBILL"
    UTILITY = "UTILITY"
    AIRTIME = "AIRTIME"


INFLOW_TYPES: frozenset[str] = frozenset(
    {TransactionType.CASH_IN.value, TransactionType.P2P_RECEIVE.value}
)

OUTFLOW_TYPES: frozenset[str] = frozenset(
    transaction_type.value
    for transaction_type in TransactionType
    if transaction_type.value not in INFLOW_TYPES
)

PRODUCTIVE_OUTFLOW_TYPES: frozenset[str] = frozenset(
    {
        TransactionType.BUY_GOODS_TILL.value,
        TransactionType.MOMOPAY_MERCHANT.value,
        TransactionType.PAYBILL.value,
        TransactionType.UTILITY.value,
    }
)

FEATURE_TRANSACTION_COLUMNS: tuple[str, ...] = (
    "applicant_id",
    "timestamp",
    "tx_type",
    "amount",
    "post_balance",
)

CANONICAL_TRANSACTION_COLUMNS: tuple[str, ...] = (
    "applicant_id",
    "timestamp",
    "provider",
    "tx_type",
    "amount",
    "post_balance",
    "transaction_id",
    "raw_text",
)


@dataclass(frozen=True)
class NormalizedTransaction:
    """Validated transaction at the ingestion/feature-engineering boundary."""

    applicant_id: str
    timestamp: datetime
    provider: TransactionProvider
    tx_type: TransactionType
    amount: float
    post_balance: float
    transaction_id: str | None = None
    raw_text: str | None = None

    def to_record(self) -> dict[str, Any]:
        """Return a DataFrame-ready canonical transaction dictionary."""
        return {
            "applicant_id": self.applicant_id,
            "timestamp": self.timestamp,
            "provider": self.provider.value,
            "tx_type": self.tx_type.value,
            "amount": self.amount,
            "post_balance": self.post_balance,
            "transaction_id": self.transaction_id,
            "raw_text": self.raw_text,
        }


def transactions_to_dataframe(
    transactions: Iterable[NormalizedTransaction],
) -> pd.DataFrame:
    """Convert canonical transactions to the feature builder's DataFrame schema."""
    records = [transaction.to_record() for transaction in transactions]
    return pd.DataFrame(records, columns=CANONICAL_TRANSACTION_COLUMNS)
