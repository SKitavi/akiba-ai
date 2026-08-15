"""Tests for SMS parser scaffolding."""

from src.ingestion.sms_parser import parse_sms_message


def test_parse_mpesa_sms() -> None:
    sms = (
        "UH13Q2B7N6 Confirmed. Ksh750.00 sent to HARUN MWANGI 0112259522 "
        "on 1/8/26 at 5:19 PM. New M-PESA balance is Ksh0.00. "
        "Transaction cost, Ksh12.00."
    )
    res = parse_sms_message(sms)
    assert res["tx_id"] == "UH13Q2B7N6"
    assert res["amount"] == 750.00
    assert res["fee"] == 12.00
    assert res["balance"] == 0.0
    assert res["counterparty"] == "HARUN MWANGI 0112259522"
    assert res["provider"] == "M-Pesa"
    assert res["timestamp"] == "2026-08-01 17:19:00"
    assert res["tx_type"] == "P2P_SEND"


def test_parse_momo_sms() -> None:
    sms = "*165*S*2500 RWF transferred to 25078123456 at 2026-03-14 14:22:10 .Fee: 10RWF.Balance: 12000RWF."
    res = parse_sms_message(sms)
    assert res["amount"] == 2500.0
    assert res["fee"] == 10.0
    assert res["balance"] == 12000.0
    assert res["counterparty"] == "25078123456"
    assert res["provider"] == "MTN_MoMo"
    assert res["tx_id"] is None
    assert res["timestamp"] == "2026-03-14 14:22:10"
    assert res["tx_type"] == "P2P_SEND"


def test_parse_receipt_format() -> None:
    receipt = """
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
"""
    res = parse_sms_message(receipt)
    assert res["tx_id"] == "31196215166"
    assert res["amount"] == 2500.0
    assert res["fee"] == 0.0
    assert res["balance"] == 14200.0
    assert res["counterparty"] == "NYARUGENGE MARKET"
    assert res["provider"] == "MTN_MoMo"
    assert res["timestamp"] == "2026-08-13 12:30:15"
    assert res["tx_type"] is None
