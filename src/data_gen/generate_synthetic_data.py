"""Generate synthetic SACCO credit-risk records.

Purpose: Define synthetic dataset generation interfaces for offline experiments.
Owner: Swafiyah (Data Engineer).
Sprint day due: Day 2 (Aug 11) - synthetic data milestone.
"""

import random
import string
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ----------------------------------------------------------------------
# GLOBAL SETUP & CONFIGURATION
# ----------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

NUM_APPLICANTS = 250
TARGET_DEFAULT_RATE = 0.20  # 20% Non-Performing Loan (NPL) rate threshold # cite: 3, 9
START_DATE = datetime(2026, 1, 1)
END_DATE = datetime(2026, 6, 30)

# Localized Biller Catalogs
UTILITIES_MPESA = [
    {"name": "KPLC PREPAID", "type": "PAYBILL", "acc_prefix": "37", "fee": 15.0},
    {"name": "NAIROBI WATER", "type": "PAYBILL", "acc_prefix": "NW-", "fee": 22.0}
]

UTILITIES_MOMO = [
    {"name": "REG (EUCL)", "ref_label": "meter", "ref_prefix": "011"},
    {"name": "WASAC", "ref_label": "account", "ref_prefix": ""}
]

MERCHANTS_MPESA_TILL = ["Naivas Supermarket", "Mama Mboga Grocery", "Kiprotich Duka", "City Pharmacy"]
MERCHANTS_MOMO_CODE = [("KIGALI STORE", "123456"), ("NYARUGENGE MARKET", "654321"), ("INYANANGE DAIRY", "112233")]

# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------
def generate_tx_id():
    """Generates a random 10-character alphanumeric transaction ID."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=10))

def generate_phone(provider):
    """Generates localized telephone numbers."""
    if provider == "M-Pesa":
        return f"07{random.randint(10000000, 99999999)}"
    return f"25078{random.randint(1000000, 9999999)}"


# ----------------------------------------------------------------------
# PROVIDER-SPECIFIC SMS FORMATTER ENGINE
# ----------------------------------------------------------------------
def build_sms_body(provider, tx_type, amount, balance, tx_id, tx_date,
                   counterparty="", account_ref="", fee=0.0):
    """
    Renders realistic, provider-specific SMS bodies# cite: 42, 43.
    """
    if provider == "M-Pesa":
        date_str = tx_date.strftime("%d/%m/%y at %I:%M %p")

        if tx_type == "P2P_SEND":
            return (f"{tx_id} Confirmed. Ksh{amount:,.2f} sent to {counterparty} "
                    f"on {date_str}. New M-PESA balance is Ksh{balance:,.2f}. "
                    f"Transaction cost, Ksh{fee:,.2f}. Download My OneApp on https://saf.cx/lPKcC")
        elif tx_type == "P2P_RECEIVE":
            return (f"{tx_id} Confirmed. You have received Ksh{amount:,.2f} from {counterparty} "
                    f"on {date_str}. New M-PESA balance is Ksh{balance:,.2f}.")
        elif tx_type == "CASH_IN":
            return (f"{tx_id} Confirmed. Received Ksh{amount:,.2f} from Agent {counterparty} "
                    f"on {date_str}. New M-PESA balance is Ksh{balance:,.2f}.")
        elif tx_type == "CASH_OUT":
            return (f"{tx_id} Confirmed. Ksh{amount:,.2f} withdrawn from Agent {counterparty} "
                    f"on {date_str}. New M-PESA balance is Ksh{balance:,.2f}. Transaction cost, Ksh{fee:,.2f}.")
        elif tx_type == "BUY_GOODS_TILL":
            return (f"{tx_id} Confirmed. Ksh{amount:,.2f} paid to {counterparty} "
                    f"on {date_str}. New M-PESA balance is Ksh{balance:,.2f}. Transaction cost, Ksh0.00.")
        elif tx_type == "PAYBILL":
            return (f"{tx_id} Confirmed. Ksh{amount:,.2f} sent to {counterparty} AC "
                    f"for account {account_ref} on {date_str}. "
                    f"New M-PESA balance is Ksh{balance:,.2f}. Transaction cost, Ksh{fee:,.2f}.")
        elif tx_type == "AIRTIME":
            return (f"{tx_id} Confirmed. Ksh{amount:,.2f} paid for Airtime "
                    f"on {date_str}. New M-PESA balance is Ksh{balance:,.2f}. Transaction cost, Ksh0.00.")

    elif provider == "MTN_MoMo":
        date_str = tx_date.strftime("%Y-%m-%d %H:%M:%S")
        amt_i, bal_i, fee_i = int(amount), int(balance), int(fee)

        if tx_type == "P2P_SEND":
            return (f"*165*S*{amt_i} RWF transferred to {counterparty} "
                    f"at {date_str} .Fee: {fee_i}RWF.Balance: {bal_i}RWF."
                    f"Dial *182*1*3# and send money abroad *EN#")
        elif tx_type == "P2P_RECEIVE":
            return (f"You have received {amt_i} RWF from {counterparty} "
                    f"at {date_str}. Balance: {bal_i}RWF.")
        elif tx_type == "CASH_IN":
            return (f"TxID:{tx_id} Cash In of {amt_i} RWF from Agent {counterparty} "
                    f"at {date_str} .Fee: 0RWF.Balance: {bal_i}RWF.")
        elif tx_type == "CASH_OUT":
            return (f"TxID:{tx_id} Cash Out of {amt_i} RWF from Agent {counterparty} "
                    f"at {date_str} .Fee: {fee_i}RWF.Balance: {bal_i}RWF.")
        elif tx_type == "MOMOPAY_MERCHANT":
            code_str = f"({account_ref}) " if account_ref else ""
            return (f"TxID:{tx_id} Payment of {amt_i} RWF to {counterparty} {code_str}"
                    f"successful at {date_str} .Fee: {fee_i}RWF.Balance: {bal_i}RWF.")
        elif tx_type == "UTILITY":
            return (f"TxID:{tx_id} Payment of {amt_i} RWF to {counterparty} "
                    f"for {account_ref} successful at {date_str} .Fee: 0RWF.Balance: {bal_i}RWF.")
        elif tx_type == "AIRTIME":
            return (f"You bought {amt_i} RWF of Airtime at {date_str}. Balance: {bal_i}RWF. Enjoy MTN services.")

    return f"TxID:{tx_id} Amount: {amount}. Balance: {balance}."

# ----------------------------------------------------------------------
# APPLICANT GENERATION & TRANSACTION LOGIC
# ----------------------------------------------------------------------
def generate_dataset(num_applicants=NUM_APPLICANTS):
    applicants = []
    sms_logs = []

    for i in range(1, num_applicants + 1):
        app_id = f"APP_{i:04d}"
        persona = random.choice(["informal_trader", "farmer"])  # cite: 13, 14
        provider = random.choice(["M-Pesa", "MTN_MoMo"])

        # Financial behavior parameters tuned by persona # cite: 3, 4, 6
        if persona == "informal_trader":
            income = np.random.uniform(150000, 600000)
            monthly_txs = random.randint(30, 75)  # High daily cash flow frequency # cite: 4
        else:  # farmer
            income = np.random.uniform(100000, 800000)
            monthly_txs = random.randint(6, 22)   # Lumpy, seasonal transaction pattern # cite: 6

        total_days = (END_DATE - START_DATE).days
        total_tx_count = int(monthly_txs * (total_days / 30.0))
        random_days = np.sort(np.random.randint(0, total_days, size=total_tx_count))

        current_balance = np.random.uniform(2000, 30000)

        for day_offset in random_days:
            tx_date = START_DATE + timedelta(days=int(day_offset),
                                             hours=random.randint(6, 21),
                                             minutes=random.randint(0, 59))

            # Select Transaction Type
            tx_type = random.choices(
                ["CASH_IN", "P2P_RECEIVE", "CASH_OUT", "P2P_SEND", "MERCHANT", "UTILITY", "AIRTIME"],
                weights=[0.20, 0.25, 0.20, 0.15, 0.10, 0.05, 0.05]
            )[0]

            fee = 0.0
            counterparty = ""
            account_ref = ""

            # Money In
            if tx_type in ["CASH_IN", "P2P_RECEIVE"]:
                amount = round(np.random.exponential(scale=income / 8), -2)
                amount = max(200, min(amount, 250000))
                current_balance += amount
                counterparty = f"Agent_{random.randint(100, 999)}" if tx_type == "CASH_IN" else generate_phone(provider)

            # Money Out
            else:
                amount = round(np.random.exponential(scale=income / 12), -2)
                amount = max(100, min(amount, current_balance * 0.95))
                if current_balance < amount:
                    amount = max(50, current_balance * 0.5)

                current_balance -= amount

                if tx_type == "CASH_OUT":
                    fee = 20.0 if provider == "MTN_MoMo" else 13.0
                    counterparty = f"Agent_{random.randint(100, 999)}"
                elif tx_type == "P2P_SEND":
                    fee = 10.0 if provider == "MTN_MoMo" else 12.0
                    counterparty = generate_phone(provider)
                elif tx_type == "MERCHANT":
                    if provider == "M-Pesa":
                        tx_type = "BUY_GOODS_TILL"
                        counterparty = random.choice(MERCHANTS_MPESA_TILL)
                    else:
                        tx_type = "MOMOPAY_MERCHANT"
                        m_name, m_code = random.choice(MERCHANTS_MOMO_CODE)
                        counterparty, account_ref = m_name, m_code
                elif tx_type == "UTILITY":
                    if provider == "M-Pesa":
                        tx_type = "PAYBILL"
                        util = random.choice(UTILITIES_MPESA)
                        counterparty = util["name"]
                        account_ref = f"{util['acc_prefix']}{random.randint(100000, 999999)}"
                        fee = util["fee"]
                    else:
                        util = random.choice(UTILITIES_MOMO)
                        counterparty = util["name"]
                        account_ref = f"{util['ref_label']} {util['ref_prefix']}{random.randint(100000, 999999)}"
                elif tx_type == "AIRTIME":
                    counterparty = "Self"

            # Render provider-specific SMS body
            tx_id = generate_tx_id()
            sms_body = build_sms_body(
                provider=provider, tx_type=tx_type, amount=amount,
                balance=current_balance, tx_id=tx_id, tx_date=tx_date,
                counterparty=counterparty, account_ref=account_ref, fee=fee
            )

            sms_logs.append({
                "applicant_id": app_id,
                "timestamp": tx_date.strftime("%Y-%m-%d %H:%M:%S"),
                "provider": provider,
                "sms_text": sms_body,
                "tx_type": tx_type,
                "amount": amount,
                "post_balance": current_balance
            })

        applicants.append({
            "applicant_id": app_id,
            "persona": persona,
            "provider": provider,
            "avg_monthly_income": income
        })

    return pd.DataFrame(applicants), pd.DataFrame(sms_logs)

# ----------------------------------------------------------------------
# EXECUTION & RISK LABEL CALIBRATION
# ----------------------------------------------------------------------
def main():
    print("1. Generating Base Applicant Cohort & SMS Logs...")
    df_applicants, df_sms = generate_dataset(NUM_APPLICANTS)

    print("2. Computing Ground-Truth Risk Indicators...")
    applicant_metrics = []

    for app_id, group in df_sms.groupby("applicant_id"):
        inflows = group[group["tx_type"].isin(["CASH_IN", "P2P_RECEIVE"])]["amount"].sum()
        outflows = group[group["tx_type"].isin(["CASH_OUT", "P2P_SEND", "BUY_GOODS_TILL", "MOMOPAY_MERCHANT", "PAYBILL", "UTILITY", "AIRTIME"])]["amount"].sum()
        net_flow = inflows - outflows
        tx_count = len(group)
        low_balance_events = (group["post_balance"] < 1500).sum()

        persona_row = df_applicants.loc[df_applicants["applicant_id"] == app_id]
        persona = persona_row["persona"].values[0]
        income = persona_row["avg_monthly_income"].values[0]

        # Compute net flow relative to baseline monthly income
        net_flow_ratio = net_flow / income if income > 0 else 0.0

        # Mathematical Ground-Truth Risk Score using relative ratio features # cite: 8, 28, 29
        risk_score = (
            (-3.0 * net_flow_ratio) +
            (0.20 * low_balance_events) +
            (-0.015 * tx_count) +
            (0.6 if persona == "farmer" and net_flow < 0 else 0) +
            np.random.normal(0, 0.4)
        )

        applicant_metrics.append({
            "applicant_id": app_id,
            "net_flow": net_flow,
            "tx_count": tx_count,
            "low_balance_events": low_balance_events,
            "raw_risk_score": risk_score
        })

    df_metrics = pd.DataFrame(applicant_metrics)

    # Calibrate Target Non-Performing Loan (Default) Rate ~ 20% # cite: 3, 9, 30, 31
    threshold = df_metrics["raw_risk_score"].quantile(1 - TARGET_DEFAULT_RATE)
    df_metrics["default_label"] = (df_metrics["raw_risk_score"] >= threshold).astype(int)

    # Combine into final applicant target frame
    df_final_applicants = pd.merge(df_applicants, df_metrics[["applicant_id", "net_flow", "tx_count", "low_balance_events", "default_label"]], on="applicant_id")

    # Save Datasets to Disk
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "data" / "raw"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df_sms.to_csv(DATA_DIR / "synthetic_momo_sms_logs.csv", index=False)
    df_final_applicants.to_csv(DATA_DIR / "synthetic_applicants_labeled.csv", index=False)

    # ----------------------------------------------------------------------
    # DATASET SUMMARY REPORT
    # ----------------------------------------------------------------------
    print("\n" + "="*50)
    print("     SYNTHETIC DATASET GENERATION SUMMARY")
    print("="*50)
    print(f"Total Applicants Labeled: {len(df_final_applicants)}")
    print(f"Total SMS Logs Generated : {len(df_sms)}")
    print(f"Calibrated Default Rate  : {df_final_applicants['default_label'].mean() * 100:.2f}%")
    print("-" * 50)
    print("Default Rate Distribution by Persona:")
    print(pd.crosstab(df_final_applicants["persona"], df_final_applicants["default_label"], normalize="index").round(3) * 100)
    print("-" * 50)
    print("Sample Generated SMS Receipts:")
    for sample_sms in df_sms["sms_text"].sample(3, random_state=SEED):
        print(f" > {sample_sms}")
    print("="*50)


if __name__ == "__main__":
    main()