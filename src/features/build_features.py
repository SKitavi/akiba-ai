"""Behavioral feature engineering for Akiba AI credit scoring.

Transforms per-applicant SMS transaction logs into a model-ready feature table.
All features are currency-agnostic (ratio/relative) to handle KES vs RWF safely.

Feature groups produced:
  - Velocity        : transaction rate and temporal patterns
  - Net-flow        : cash-flow level and stability (CV)
  - Income proxy    : inflow mean, std, regularity
  - Expense proxy   : outflow mean, std, dominant spend category
  - Balance health  : low-balance events, min balance, balance trend
  - Frequency mix   : productive vs cash-only transaction ratio

Input DataFrame expected columns (from generate_synthetic_data or sms_parser pipeline):
    applicant_id, timestamp, tx_type, amount, post_balance

Optional column (used when joining applicant metadata):
    avg_monthly_income  (used to compute net_flow_ratio; defaults to inflow sum if absent)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

# ---------------------------------------------------------------------------
# Transaction type category maps
# ---------------------------------------------------------------------------

#: Transaction types that represent money flowing IN to the wallet
INFLOW_TYPES: frozenset[str] = frozenset({"CASH_IN", "P2P_RECEIVE"})

#: Transaction types that represent money flowing OUT of the wallet
OUTFLOW_TYPES: frozenset[str] = frozenset(
    {
        "CASH_OUT",
        "P2P_SEND",
        "BUY_GOODS_TILL",
        "MOMOPAY_MERCHANT",
        "PAYBILL",
        "UTILITY",
        "AIRTIME",
    }
)

#: "Productive" outflows (bills, goods, utilities) — less risky than pure cash-out
PRODUCTIVE_OUTFLOW_TYPES: frozenset[str] = frozenset(
    {"BUY_GOODS_TILL", "MOMOPAY_MERCHANT", "PAYBILL", "UTILITY"}
)

#: Low-balance threshold (consistent with calibrate_default_labels in generate_synthetic_data)
LOW_BALANCE_THRESHOLD: float = 1500.0


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _observation_months(group: pd.DataFrame) -> float:
    """Return the number of months spanned by the group's timestamps.

    Minimum of 1 to avoid division-by-zero for applicants with very few events.
    """
    timestamps = pd.to_datetime(group["timestamp"])
    span_days = (timestamps.max() - timestamps.min()).days
    return max(span_days / 30.0, 1.0)


def _cv(series: pd.Series) -> float:
    """Coefficient of variation (std / mean). Returns 0 if mean is zero."""
    mean = series.mean()
    if mean == 0 or len(series) < 2:
        return 0.0
    return float(series.std(ddof=1) / abs(mean))


# ---------------------------------------------------------------------------
# Per-applicant feature computation
# ---------------------------------------------------------------------------

def _compute_applicant_features(app_id: str, group: pd.DataFrame, income: float) -> dict:
    """Compute all behavioral features for a single applicant.

    Args:
        app_id:  Applicant identifier string.
        group:   Subset of the events DataFrame for this applicant.
        income:  Baseline monthly income estimate (used for ratio normalisation).
                 Derived from avg_monthly_income if available, else falls back to
                 observed inflow sum.

    Returns:
        A flat dict of feature values keyed by feature name.
    """
    group = group.copy()
    group["timestamp"] = pd.to_datetime(group["timestamp"])
    group = group.sort_values("timestamp")

    obs_months = _observation_months(group)
    total_tx = len(group)

    inflows = group[group["tx_type"].isin(INFLOW_TYPES)]
    outflows = group[group["tx_type"].isin(OUTFLOW_TYPES)]
    productive = group[group["tx_type"].isin(PRODUCTIVE_OUTFLOW_TYPES)]

    # ------------------------------------------------------------------
    # 1. VELOCITY FEATURES
    # ------------------------------------------------------------------
    tx_per_month = total_tx / obs_months

    # Daily transaction counts → weekly peak
    group["date"] = group["timestamp"].dt.date
    daily_counts = group.groupby("date").size()
    # Aggregate to weekly buckets
    group["week"] = group["timestamp"].dt.isocalendar().week.astype(int)
    weekly_counts = group.groupby("week").size()
    peak_week_tx = int(weekly_counts.max()) if not weekly_counts.empty else 0

    # Recency: days since last transaction (relative to observation window end)
    obs_end = group["timestamp"].max()
    days_since_last_tx = float((obs_end - group["timestamp"].max()).days)  # 0 within window
    # More useful: days since last INFLOW
    if not inflows.empty:
        days_since_last_inflow = float((obs_end - inflows["timestamp"].max()).days)
    else:
        days_since_last_inflow = float(obs_months * 30)  # treat as full period absent

    # Active months: number of distinct calendar months with ≥1 transaction
    group["month"] = group["timestamp"].dt.to_period("M")
    active_months = group["month"].nunique()

    # ------------------------------------------------------------------
    # 2. NET CASH-FLOW STABILITY
    # ------------------------------------------------------------------
    # Monthly net flows
    group["month_str"] = group["timestamp"].dt.to_period("M").astype(str)
    monthly_inflow = (
        inflows.assign(month_str=inflows["timestamp"].dt.to_period("M").astype(str))
        .groupby("month_str")["amount"]
        .sum()
        .reindex(group["month_str"].unique(), fill_value=0.0)
    )
    monthly_outflow = (
        outflows.assign(month_str=outflows["timestamp"].dt.to_period("M").astype(str))
        .groupby("month_str")["amount"]
        .sum()
        .reindex(group["month_str"].unique(), fill_value=0.0)
    )
    monthly_net = monthly_inflow - monthly_outflow

    net_flow_total = float(monthly_net.sum())
    net_flow_mean = float(monthly_net.mean())
    net_flow_std = float(monthly_net.std(ddof=1)) if len(monthly_net) > 1 else 0.0
    net_flow_cv = _cv(monthly_net)

    # Ratio of net flow to income baseline (currency-agnostic)
    income_baseline = income if income > 0 else 1.0
    net_flow_ratio = net_flow_total / income_baseline

    # Months with negative net flow
    negative_net_months = int((monthly_net < 0).sum())

    # ------------------------------------------------------------------
    # 3. INFLOW (INCOME PROXY) FEATURES
    # ------------------------------------------------------------------
    if not inflows.empty:
        inflow_total = float(inflows["amount"].sum())
        inflow_mean = float(inflows["amount"].mean())
        inflow_std = float(inflows["amount"].std(ddof=1)) if len(inflows) > 1 else 0.0
        inflow_cv = _cv(inflows["amount"])
        inflow_count = len(inflows)
        inflow_per_month = inflow_count / obs_months

        # Regularity: fraction of active months that had ≥1 inflow event
        months_with_inflow = (
            inflows.assign(month_str=inflows["timestamp"].dt.to_period("M").astype(str))["month_str"]
            .nunique()
        )
        inflow_regularity = months_with_inflow / max(active_months, 1)
    else:
        inflow_total = inflow_mean = inflow_std = inflow_cv = 0.0
        inflow_count = 0
        inflow_per_month = 0.0
        inflow_regularity = 0.0

    # ------------------------------------------------------------------
    # 4. OUTFLOW (EXPENSE) FEATURES
    # ------------------------------------------------------------------
    if not outflows.empty:
        outflow_total = float(outflows["amount"].sum())
        outflow_mean = float(outflows["amount"].mean())
        outflow_std = float(outflows["amount"].std(ddof=1)) if len(outflows) > 1 else 0.0
        outflow_cv = _cv(outflows["amount"])
        outflow_count = len(outflows)
        outflow_per_month = outflow_count / obs_months
        productive_ratio = len(productive) / outflow_count
    else:
        outflow_total = outflow_mean = outflow_std = outflow_cv = 0.0
        outflow_count = 0
        outflow_per_month = 0.0
        productive_ratio = 0.0

    # Ratio of inflow to outflow (>1 = healthy, <1 = distress)
    inflow_outflow_ratio = inflow_total / max(outflow_total, 1.0)

    # ------------------------------------------------------------------
    # 5. BALANCE HEALTH FEATURES
    # ------------------------------------------------------------------
    balances = group["post_balance"]
    low_balance_events = int((balances < LOW_BALANCE_THRESHOLD).sum())
    min_balance = float(balances.min())
    mean_balance = float(balances.mean())

    # Balance trend: slope of a simple linear fit over time index
    if len(balances) >= 3:
        x = np.arange(len(balances), dtype=float)
        slope, _ = np.polyfit(x, balances.values, 1)
        balance_trend_slope = float(slope)
    else:
        balance_trend_slope = 0.0

    # Normalised low-balance rate (events per month)
    low_balance_rate = low_balance_events / obs_months

    # ------------------------------------------------------------------
    # 6. TRANSACTION TYPE MIX
    # ------------------------------------------------------------------
    type_counts = group["tx_type"].value_counts()
    airtime_ratio = type_counts.get("AIRTIME", 0) / max(total_tx, 1)
    cashout_ratio = type_counts.get("CASH_OUT", 0) / max(total_tx, 1)
    p2p_send_ratio = type_counts.get("P2P_SEND", 0) / max(total_tx, 1)
    p2p_receive_ratio = type_counts.get("P2P_RECEIVE", 0) / max(total_tx, 1)

    return {
        "applicant_id": app_id,
        # --- Velocity ---
        "tx_per_month": round(tx_per_month, 4),
        "peak_week_tx": peak_week_tx,
        "active_months": active_months,
        "days_since_last_inflow": round(days_since_last_inflow, 1),
        # --- Net cash-flow stability ---
        "net_flow_total": round(net_flow_total, 2),
        "net_flow_mean": round(net_flow_mean, 2),
        "net_flow_std": round(net_flow_std, 2),
        "net_flow_cv": round(net_flow_cv, 4),
        "net_flow_ratio": round(net_flow_ratio, 6),
        "negative_net_months": negative_net_months,
        # --- Inflow (income proxy) ---
        "inflow_total": round(inflow_total, 2),
        "inflow_mean": round(inflow_mean, 2),
        "inflow_std": round(inflow_std, 2),
        "inflow_cv": round(inflow_cv, 4),
        "inflow_per_month": round(inflow_per_month, 4),
        "inflow_regularity": round(inflow_regularity, 4),
        # --- Outflow (expense proxy) ---
        "outflow_total": round(outflow_total, 2),
        "outflow_mean": round(outflow_mean, 2),
        "outflow_std": round(outflow_std, 2),
        "outflow_cv": round(outflow_cv, 4),
        "outflow_per_month": round(outflow_per_month, 4),
        "productive_ratio": round(productive_ratio, 4),
        "inflow_outflow_ratio": round(inflow_outflow_ratio, 4),
        # --- Balance health ---
        "low_balance_events": low_balance_events,
        "low_balance_rate": round(low_balance_rate, 4),
        "min_balance": round(min_balance, 2),
        "mean_balance": round(mean_balance, 2),
        "balance_trend_slope": round(balance_trend_slope, 4),
        # --- Tx type mix ---
        "airtime_ratio": round(airtime_ratio, 4),
        "cashout_ratio": round(cashout_ratio, 4),
        "p2p_send_ratio": round(p2p_send_ratio, 4),
        "p2p_receive_ratio": round(p2p_receive_ratio, 4),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feature_table(
    events_df: pd.DataFrame,
    applicants_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Transform normalized transaction events into a model-ready feature table.

    Aggregates per-transaction SMS log rows into one feature row per applicant
    covering velocity, net cash-flow stability, income proxy, expense proxy,
    balance health, and transaction-type mix.

    Args:
        events_df:     DataFrame of transaction events. Required columns:
                           ``applicant_id``, ``timestamp``, ``tx_type``,
                           ``amount``, ``post_balance``.
                       Optional column: ``avg_monthly_income`` (if present and
                       ``applicants_df`` is not provided, it is used directly).
        applicants_df: Optional applicant metadata DataFrame containing
                       ``applicant_id`` and ``avg_monthly_income``. When
                       provided, income is joined from this frame; when absent
                       the function falls back to ``events_df`` if it carries
                       the column, otherwise uses observed inflow sum.

    Returns:
        DataFrame with one row per applicant and all computed feature columns.
        Sorted by ``applicant_id``.

    Raises:
        ValueError: If required columns are missing from ``events_df``.
    """
    required = {"applicant_id", "timestamp", "tx_type", "amount", "post_balance"}
    missing = required - set(events_df.columns)
    if missing:
        raise ValueError(f"events_df is missing required columns: {missing}")

    # Build income lookup: applicant_id → avg_monthly_income
    income_map: dict[str, float] = {}
    if applicants_df is not None and "avg_monthly_income" in applicants_df.columns:
        income_map = dict(
            zip(applicants_df["applicant_id"], applicants_df["avg_monthly_income"])
        )
    elif "avg_monthly_income" in events_df.columns:
        # May be repeated per row (joined upstream) — take max per applicant
        income_map = (
            events_df.groupby("applicant_id")["avg_monthly_income"].max().to_dict()
        )

    records = []
    for app_id, group in events_df.groupby("applicant_id", sort=True):
        # Resolve income baseline
        if app_id in income_map:
            income = float(income_map[app_id])
        else:
            # Fallback: use total observed inflows as income proxy
            inflow_sum = group[group["tx_type"].isin(INFLOW_TYPES)]["amount"].sum()
            income = float(inflow_sum) if inflow_sum > 0 else 1.0

        records.append(_compute_applicant_features(str(app_id), group, income))

    return pd.DataFrame(records).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature column list (useful for model training / selection)
# ---------------------------------------------------------------------------

#: Ordered list of numeric feature columns produced by ``build_feature_table``
#: (excludes the ``applicant_id`` identifier column).
FEATURE_COLUMNS: list[str] = [
    # Velocity
    "tx_per_month",
    "peak_week_tx",
    "active_months",
    "days_since_last_inflow",
    # Net cash-flow stability
    "net_flow_total",
    "net_flow_mean",
    "net_flow_std",
    "net_flow_cv",
    "net_flow_ratio",
    "negative_net_months",
    # Inflow (income proxy)
    "inflow_total",
    "inflow_mean",
    "inflow_std",
    "inflow_cv",
    "inflow_per_month",
    "inflow_regularity",
    # Outflow (expense proxy)
    "outflow_total",
    "outflow_mean",
    "outflow_std",
    "outflow_cv",
    "outflow_per_month",
    "productive_ratio",
    "inflow_outflow_ratio",
    # Balance health
    "low_balance_events",
    "low_balance_rate",
    "min_balance",
    "mean_balance",
    "balance_trend_slope",
    # Tx type mix
    "airtime_ratio",
    "cashout_ratio",
    "p2p_send_ratio",
    "p2p_receive_ratio",
]
