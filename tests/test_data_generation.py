"""Tests for synthetic data generation engine."""

import pytest
import pandas as pd
from src.data_gen.generate_synthetic_data import generate_dataset, generate_phone, generate_tx_id


def test_generate_phone_formats() -> None:
    """Verifies that generated telephone numbers adhere to provider format standards."""
    # M-Pesa phone numbers should be 10 characters long and start with '07'
    mpesa_phone = generate_phone("M-Pesa")
    assert len(mpesa_phone) == 10
    assert mpesa_phone.startswith("07")
    assert mpesa_phone.isdigit()

    # MTN MoMo phone numbers should be 12 characters long and start with '25078'
    momo_phone = generate_phone("MTN_MoMo")
    assert len(momo_phone) == 12
    assert momo_phone.startswith("25078")
    assert momo_phone.isdigit()


def test_generate_tx_id_format() -> None:
    """Verifies that generated transaction IDs are 10-character alphanumeric strings."""
    tx_id = generate_tx_id()
    assert len(tx_id) == 10
    assert tx_id.isalnum()
    assert tx_id.isupper()


def test_generate_dataset_structure() -> None:
    """Smoke test to verify columns, shapes, and types in the generated dataset."""
    num_test_applicants = 10
    df_applicants, df_sms = generate_dataset(num_test_applicants)

    # Verify return types
    assert isinstance(df_applicants, pd.DataFrame)
    assert isinstance(df_sms, pd.DataFrame)

    # Verify applicant row count
    assert len(df_applicants) == num_test_applicants

    # Verify applicant columns
    expected_app_cols = {"applicant_id", "persona", "provider", "avg_monthly_income"}
    assert expected_app_cols.issubset(df_applicants.columns)

    # Verify SMS log columns
    expected_sms_cols = {
        "applicant_id",
        "timestamp",
        "provider",
        "sms_text",
        "tx_type",
        "amount",
        "post_balance"
    }
    assert expected_sms_cols.issubset(df_sms.columns)

    # Verify relationship integrity
    app_ids = set(df_applicants["applicant_id"])
    sms_app_ids = set(df_sms["applicant_id"])
    assert sms_app_ids.issubset(app_ids)
