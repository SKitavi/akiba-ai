# Mobile Money Synthetic Data Schema & Distribution Specification

## Overview
This document defines the schema definitions, field metadata, and behavioral distribution rules for the synthetic mobile-money transaction dataset (M-Pesa and MTN MoMo formats).

---

## Data Schemas

### 1. Raw SMS Transaction Logs (`synthetic_momo_sms_logs.csv`)
Contains raw and parsed mobile money transaction receipts generated for synthetic applicants.

| Field Name | Data Type | Key Type | Description | Example / Allowed Values |
| :--- | :--- | :--- | :--- | :--- |
| `applicant_id` | String | Foreign Key | Unique identifier assigned to borrower | `APP_0001` |
| `timestamp` | Datetime | - | Date and time of the transaction (`YYYY-MM-DD HH:MM:SS`) | `2026-03-14 14:22:10` |
| `provider` | Categorical | - | Mobile Network Operator / Gateway | `M-Pesa`, `MTN_MoMo` |
| `tx_type` | Categorical | - | Transaction classification | `P2P_SEND`, `P2P_RECEIVE`, `CASH_IN`, `CASH_OUT`, `BUY_GOODS_TILL`, `MOMOPAY_MERCHANT`, `PAYBILL`, `UTILITY`, `AIRTIME` |
| `amount` | Float | - | Nominal transaction value | `750.00` (KES) / `2500` (RWF) |
| `post_balance` | Float | - | Wallet balance immediately following the transaction | `1420.00` |
| `sms_text` | Text | - | Raw provider-authentic SMS message body (includes Till, Paybill, utility IDs, fees, and promo footers) | `UH13Q2B7N6 Confirmed. Ksh750.00 sent to HARUN MWANGI 0112259522 on 1/8/26 at 5:19 PM. New M-PESA balance is Ksh0.00.` |

---

### 2. Applicant Metadata & Target Labels (`synthetic_applicants_labeled.csv`)
Contains baseline economic profiles, aggregated 6-month behavioral metrics, and ground-truth credit default labels.

| Field Name | Data Type | Key Type | Description | Example / Target Value |
| :--- | :--- | :--- | :--- | :--- |
| `applicant_id` | String | Primary Key | Unique applicant identifier | `APP_0001` |
| `persona` | Categorical | - | Economic archetype | `informal_trader`, `farmer` |
| `provider` | Categorical | - | Primary MNO channel | `M-Pesa`, `MTN_MoMo` |
| `avg_monthly_income` | Float | - | Baseline estimated monthly income | `350000.0` |
| `net_flow` | Float | - | Aggregate net cash flow over the observation window (Inflow - Outflow) | `45200.0` |
| `tx_count` | Integer | - | Total count of transactions within the window | `184` |
| `low_balance_events` | Integer | - | Frequency of times wallet balance dropped below buffer threshold | `12` |
| `default_label` | Binary | **Target Label** | Ground-truth credit risk label | `0` (Non-Default / Healthy), `1` (Default / High Risk) |

---

## Behavioral Distributions & Persona Rules

1. **Informal Traders**:
   * **Transaction Velocity**: 30 – 75 transactions / month (daily cash flow).
   * **Cash-Flow Pattern**: Regular, moderate-value inflows and supplier payment outflows.
   * **Risk Markers**: Abrupt drops in daily velocity, negative net flow, frequent low-balance warnings.

2. **Smallholder Farmers**:
   * **Transaction Velocity**: 6 – 22 transactions / month.
   * **Cash-Flow Pattern**: Seasonal, high-value bulk inflows (harvest cycles) followed by low activity.
   * **Risk Markers**: Missed seasonal harvest inflows, severe liquidity dry spells outside harvest periods.

---

## Target Label Calibration & Currency Safety

* **Default Rate Target**: Calibrated to **20%** (within the proposal's 15–30% Non-Performing Loan target range) using quantile cutoff scoring on behavioral indicators.
* **Multi-Currency Safety**: Credit risk calculations utilize relative ratio features (e.g., net flow relative to baseline income, low-balance event frequency) to prevent exchange rate distortions between KES and RWF.
