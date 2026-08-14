"""Tests for behavioral feature engineering (build_features.py)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.build_features import (
    FEATURE_COLUMNS,
    LOW_BALANCE_THRESHOLD,
    build_feature_table,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_events(applicant_id: str = "APP_0001") -> pd.DataFrame:
    """Minimal valid events DataFrame for one applicant (3 months of activity)."""
    rows = [
        # Month 1: inflow then outflow
        {"applicant_id": applicant_id, "timestamp": "2026-01-05 09:00:00",
         "tx_type": "CASH_IN", "amount": 50000.0, "post_balance": 52000.0},
        {"applicant_id": applicant_id, "timestamp": "2026-01-10 14:00:00",
         "tx_type": "P2P_SEND", "amount": 10000.0, "post_balance": 42000.0},
        {"applicant_id": applicant_id, "timestamp": "2026-01-20 11:00:00",
         "tx_type": "BUY_GOODS_TILL", "amount": 5000.0, "post_balance": 37000.0},
        # Month 2: inflow then cash-out
        {"applicant_id": applicant_id, "timestamp": "2026-02-03 08:30:00",
         "tx_type": "P2P_RECEIVE", "amount": 60000.0, "post_balance": 97000.0},
        {"applicant_id": applicant_id, "timestamp": "2026-02-15 16:00:00",
         "tx_type": "CASH_OUT", "amount": 15000.0, "post_balance": 82000.0},
        {"applicant_id": applicant_id, "timestamp": "2026-02-28 10:00:00",
         "tx_type": "PAYBILL", "amount": 3000.0, "post_balance": 79000.0},
        # Month 3: small airtime + low balance event
        {"applicant_id": applicant_id, "timestamp": "2026-03-07 13:00:00",
         "tx_type": "AIRTIME", "amount": 500.0, "post_balance": 1200.0},  # below threshold
        {"applicant_id": applicant_id, "timestamp": "2026-03-25 09:00:00",
         "tx_type": "CASH_IN", "amount": 40000.0, "post_balance": 41200.0},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Schema and output structure
# ---------------------------------------------------------------------------

def test_returns_dataframe() -> None:
    df = build_feature_table(_make_events())
    assert isinstance(df, pd.DataFrame)


def test_one_row_per_applicant() -> None:
    events = pd.concat([_make_events("APP_0001"), _make_events("APP_0002")], ignore_index=True)
    result = build_feature_table(events)
    assert len(result) == 2
    assert set(result["applicant_id"]) == {"APP_0001", "APP_0002"}


def test_all_feature_columns_present() -> None:
    result = build_feature_table(_make_events())
    for col in FEATURE_COLUMNS:
        assert col in result.columns, f"Missing feature column: {col}"


def test_no_null_values_in_output() -> None:
    result = build_feature_table(_make_events())
    assert result[FEATURE_COLUMNS].isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# Velocity features
# ---------------------------------------------------------------------------

def test_tx_per_month_positive() -> None:
    result = build_feature_table(_make_events())
    assert result.loc[0, "tx_per_month"] > 0


def test_active_months_correct() -> None:
    result = build_feature_table(_make_events())
    # Events span Jan, Feb, Mar → 3 distinct calendar months
    assert result.loc[0, "active_months"] == 3


def test_peak_week_tx_at_least_one() -> None:
    result = build_feature_table(_make_events())
    assert result.loc[0, "peak_week_tx"] >= 1


def test_days_since_last_inflow_zero_for_recent_inflow() -> None:
    """Last event is a CASH_IN, so days_since_last_inflow should be 0."""
    result = build_feature_table(_make_events())
    assert result.loc[0, "days_since_last_inflow"] == 0.0


# ---------------------------------------------------------------------------
# Net cash-flow stability features
# ---------------------------------------------------------------------------

def test_net_flow_total_sign() -> None:
    """Total inflows > total outflows → positive net flow."""
    events = _make_events()
    inflow = events[events["tx_type"].isin({"CASH_IN", "P2P_RECEIVE"})]["amount"].sum()
    outflow = events[~events["tx_type"].isin({"CASH_IN", "P2P_RECEIVE"})]["amount"].sum()
    result = build_feature_table(events)
    expected_sign = 1 if inflow > outflow else -1
    actual = result.loc[0, "net_flow_total"]
    assert (actual > 0) == (expected_sign > 0)


def test_net_flow_cv_non_negative() -> None:
    result = build_feature_table(_make_events())
    assert result.loc[0, "net_flow_cv"] >= 0.0


def test_negative_net_months_count() -> None:
    """Month 3 is airtime-only (no inflow for that period in monthly grouping
    vs CASH_IN on Mar 25 — net should be positive for month 3 too).
    Validate the column is an integer and within [0, active_months]."""
    result = build_feature_table(_make_events())
    val = result.loc[0, "negative_net_months"]
    assert isinstance(val, (int, __import__("numpy").integer))
    assert 0 <= val <= result.loc[0, "active_months"]


def test_net_flow_ratio_uses_income_from_applicants_df() -> None:
    events = _make_events()
    applicants = pd.DataFrame([
        {"applicant_id": "APP_0001", "avg_monthly_income": 200000.0}
    ])
    result = build_feature_table(events, applicants_df=applicants)
    inflow_sum = events[events["tx_type"].isin({"CASH_IN", "P2P_RECEIVE"})]["amount"].sum()
    outflow_sum = events[~events["tx_type"].isin({"CASH_IN", "P2P_RECEIVE"})]["amount"].sum()
    expected_ratio = (inflow_sum - outflow_sum) / 200000.0
    assert abs(result.loc[0, "net_flow_ratio"] - expected_ratio) < 0.001


# ---------------------------------------------------------------------------
# Balance health features
# ---------------------------------------------------------------------------

def test_low_balance_events_detected() -> None:
    """The Mar 7 record has post_balance=1200 which is below LOW_BALANCE_THRESHOLD."""
    result = build_feature_table(_make_events())
    assert result.loc[0, "low_balance_events"] >= 1


def test_min_balance_is_minimum_of_post_balances() -> None:
    events = _make_events()
    expected_min = events["post_balance"].min()
    result = build_feature_table(events)
    assert abs(result.loc[0, "min_balance"] - expected_min) < 0.01


def test_balance_trend_slope_is_float() -> None:
    result = build_feature_table(_make_events())
    assert isinstance(result.loc[0, "balance_trend_slope"], float)


# ---------------------------------------------------------------------------
# Inflow / outflow features
# ---------------------------------------------------------------------------

def test_inflow_total_matches_sum() -> None:
    events = _make_events()
    expected = events[events["tx_type"].isin({"CASH_IN", "P2P_RECEIVE"})]["amount"].sum()
    result = build_feature_table(events)
    assert abs(result.loc[0, "inflow_total"] - expected) < 0.01


def test_inflow_regularity_bounded() -> None:
    result = build_feature_table(_make_events())
    assert 0.0 <= result.loc[0, "inflow_regularity"] <= 1.0


def test_productive_ratio_bounded() -> None:
    result = build_feature_table(_make_events())
    assert 0.0 <= result.loc[0, "productive_ratio"] <= 1.0


def test_inflow_outflow_ratio_positive() -> None:
    result = build_feature_table(_make_events())
    assert result.loc[0, "inflow_outflow_ratio"] > 0.0


# ---------------------------------------------------------------------------
# Transaction mix features
# ---------------------------------------------------------------------------

def test_airtime_ratio_detected() -> None:
    result = build_feature_table(_make_events())
    assert result.loc[0, "airtime_ratio"] > 0.0


def test_ratios_do_not_exceed_one() -> None:
    result = build_feature_table(_make_events())
    for col in ["airtime_ratio", "cashout_ratio", "p2p_send_ratio", "p2p_receive_ratio",
                "productive_ratio"]:
        assert result.loc[0, col] <= 1.0, f"{col} exceeded 1.0"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_single_transaction_applicant() -> None:
    """Should not crash with only one transaction."""
    events = pd.DataFrame([{
        "applicant_id": "APP_SOLO",
        "timestamp": "2026-01-01 10:00:00",
        "tx_type": "CASH_IN",
        "amount": 10000.0,
        "post_balance": 10000.0,
    }])
    result = build_feature_table(events)
    assert len(result) == 1
    assert result.loc[0, "inflow_total"] == 10000.0


def test_missing_required_column_raises_value_error() -> None:
    events = _make_events().drop(columns=["tx_type"])
    with pytest.raises(ValueError, match="missing required columns"):
        build_feature_table(events)


def test_zero_outflow_applicant() -> None:
    """Applicant with only inflows should have productive_ratio=0 and cashout_ratio=0."""
    events = pd.DataFrame([
        {"applicant_id": "APP_INONLY", "timestamp": "2026-01-05 09:00:00",
         "tx_type": "CASH_IN", "amount": 30000.0, "post_balance": 30000.0},
        {"applicant_id": "APP_INONLY", "timestamp": "2026-02-10 09:00:00",
         "tx_type": "P2P_RECEIVE", "amount": 20000.0, "post_balance": 50000.0},
    ])
    result = build_feature_table(events)
    assert result.loc[0, "outflow_total"] == 0.0
    assert result.loc[0, "productive_ratio"] == 0.0
    assert result.loc[0, "cashout_ratio"] == 0.0
