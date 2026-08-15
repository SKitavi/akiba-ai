"""Guided applicant intake and transaction-validation workflow."""

from __future__ import annotations

from typing import Final

import pandas as pd
import streamlit as st

from src.domain.transactions import TransactionType
from src.ingestion.normalization import TransactionContext, normalize_transactions
from src.ingestion.ocr_parser import OCRExtractionError
from src.ui.components import (
    render_failure_panel,
    render_metric_rows,
    render_page_header,
    render_panel_heading,
    render_step_bar,
    render_validation_counters,
)
from src.ui.services import (
    UIInputError,
    build_feature_preview,
    demo_transactions_for_applicant,
    load_demo_dataset,
    parse_receipt_upload,
    parse_sms_records,
    read_csv_records,
)


SOURCE_OPTIONS: Final[dict[str, str]] = {
    "demo": "Use synthetic demo data",
    "csv": "Upload a transaction CSV",
    "sms": "Paste mobile-money SMS",
    "receipt": "Upload a receipt image",
}

REJECTION_ACTIONS: Final[dict[str, str]] = {
    "missing_applicant": "Add an applicant_id or use the selected member.",
    "invalid_applicant": "Use a non-empty text applicant_id.",
    "applicant_mismatch": "Make every row match the selected member.",
    "missing_provider": "Add provider as M-Pesa or MTN_MoMo.",
    "unsupported_provider": "Use M-Pesa or MTN_MoMo.",
    "missing_transaction_type": "Add a supported tx_type.",
    "unsupported_transaction_type": "Use one of the supported transaction types.",
    "missing_timestamp": "Add a transaction timestamp.",
    "invalid_timestamp": "Use ISO YYYY-MM-DD HH:MM:SS.",
    "missing_amount": "Add the transaction amount.",
    "invalid_amount": "Use a positive finite amount.",
    "missing_post_balance": "Add post_balance or balance.",
    "invalid_post_balance": "Use a non-negative finite balance.",
    "balance_conflict": "Keep only one consistent balance value.",
    "invalid_record": "Provide one structured transaction per row.",
}


def _format_number(value: object, *, decimals: int = 1) -> str:
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "Not available"


def _applicant_label(applicant_id: str, applicants: pd.DataFrame) -> str:
    row = applicants.loc[applicants["applicant_id"] == applicant_id].iloc[0]
    persona = str(row.get("persona", "member")).replace("_", " ").title()
    provider = str(row.get("provider", "Provider not set"))
    return f"{applicant_id} — {persona} · {provider}"


def _render_applicant_step() -> None:
    dataset = load_demo_dataset()
    with st.container(border=True):
        render_panel_heading("Applicant", "Required")
        mode = st.radio(
            "Applicant source",
            ("Synthetic demo member", "Enter member ID"),
            key="applicant_mode_widget",
            horizontal=True,
        )
        st.caption(
            "Synthetic members are generated locally by the repository's existing "
            "demo-data pipeline. No real customer data is included."
        )
        with st.form("applicant_form"):
            if mode == "Synthetic demo member":
                applicant_ids = dataset.applicants["applicant_id"].astype(str).tolist()
                applicant_id = st.selectbox(
                    "Demo member",
                    applicant_ids,
                    format_func=lambda value: _applicant_label(
                        value, dataset.applicants
                    ),
                    key="demo_applicant_widget",
                )
            else:
                applicant_id = st.text_input(
                    "Member ID",
                    placeholder="e.g. SACCO-00142",
                    key="custom_applicant_widget",
                ).strip()
            submitted = st.form_submit_button(
                "Continue to transactions", type="primary"
            )
        if submitted:
            if not applicant_id:
                st.error("Enter a member ID before continuing.")
                return
            st.session_state.applicant_id = applicant_id
            st.session_state.applicants_df = (
                dataset.applicants if mode == "Synthetic demo member" else None
            )
            st.session_state.assessment_step = 1
            st.session_state.last_error = None
            st.rerun()


def _receipt_type_value(label: str) -> TransactionType | None:
    return None if label == "Not specified" else TransactionType(label)


def _records_for_source(source_key: str) -> tuple[object, TransactionType | None]:
    if source_key == "demo":
        dataset = load_demo_dataset()
        return (
            demo_transactions_for_applicant(
                dataset, str(st.session_state.applicant_id)
            ),
            None,
        )
    if source_key == "csv":
        upload = st.session_state.get("csv_upload_widget")
        if upload is None:
            raise UIInputError("Choose a CSV file before validating.")
        return read_csv_records(upload.getvalue()), None
    if source_key == "sms":
        return parse_sms_records(st.session_state.get("sms_input_widget", "")), None

    upload = st.session_state.get("receipt_upload_widget")
    if upload is None:
        raise UIInputError("Choose a receipt image before validating.")
    receipt_type = _receipt_type_value(
        st.session_state.get("receipt_type_widget", "Not specified")
    )
    return parse_receipt_upload(upload.getvalue(), upload.name), receipt_type


def _validate_source(source_key: str) -> None:
    records, receipt_type = _records_for_source(source_key)
    if isinstance(records, dict):
        records = [records]
    result = normalize_transactions(
        records,
        context=TransactionContext(
            applicant_id=str(st.session_state.applicant_id), tx_type=receipt_type
        ),
    )
    preview = None
    if result.valid_count:
        preview = build_feature_preview(
            result,
            str(st.session_state.applicant_id),
            st.session_state.applicants_df,
        )
    st.session_state.source_key = source_key
    st.session_state.source_records = records
    st.session_state.normalization_result = result
    st.session_state.feature_preview = preview
    st.session_state.assessment_step = 2
    st.session_state.last_error = None


def _render_source_input(source_key: str) -> None:
    if source_key == "demo":
        if st.session_state.applicants_df is None:
            st.info(
                "Demo transactions are available only for a synthetic demo member. "
                "Choose CSV, SMS, or receipt for a custom member ID."
            )
        else:
            st.caption(
                "Uses the selected member's existing synthetic transaction history."
            )
    elif source_key == "csv":
        st.file_uploader(
            "Transaction CSV",
            type=["csv"],
            key="csv_upload_widget",
            help="Rows are checked by the canonical transaction normalizer.",
        )
        with st.expander("Required CSV fields"):
            st.code(
                "timestamp,provider,tx_type,amount,post_balance\n"
                "2026-08-01 09:30:00,M-Pesa,CASH_IN,2500,8400",
                language="csv",
            )
            st.caption(
                "applicant_id is optional because the selected member supplies it."
            )
    elif source_key == "sms":
        st.text_area(
            "Mobile-money messages",
            height=180,
            placeholder="Paste a supported M-Pesa or MTN MoMo message here…",
            key="sms_input_widget",
            help="Separate multiple messages with a blank line.",
        )
    else:
        st.file_uploader(
            "Receipt image",
            type=["png", "jpg", "jpeg"],
            key="receipt_upload_widget",
        )
        st.selectbox(
            "Transaction type (only if the receipt omits it)",
            ("Not specified",) + tuple(item.value for item in TransactionType),
            key="receipt_type_widget",
        )
        st.caption(
            "OCR runs locally. AkibaAI reports setup or extraction failures and "
            "does not substitute mock receipt text."
        )


def _render_transaction_step() -> None:
    with st.container(border=True):
        render_panel_heading(
            "Transaction evidence", f"Member {st.session_state.applicant_id}"
        )
        source_label = st.radio(
            "Evidence source",
            tuple(SOURCE_OPTIONS.values()),
            key="source_widget",
            captions=(
                "Fast local walkthrough",
                "Structured export",
                "Provider messages",
                "Local OCR extraction",
            ),
        )
        source_key = next(
            key for key, label in SOURCE_OPTIONS.items() if label == source_label
        )
        _render_source_input(source_key)

        previous, validate = st.columns([1, 3])
        with previous:
            if st.button("Back", use_container_width=True):
                st.session_state.assessment_step = 0
                st.session_state.last_error = None
                st.rerun()
        with validate:
            unavailable_demo = (
                source_key == "demo" and st.session_state.applicants_df is None
            )
            if st.button(
                "Validate transactions",
                type="primary",
                use_container_width=True,
                disabled=unavailable_demo,
            ):
                try:
                    _validate_source(source_key)
                except (
                    UIInputError,
                    OCRExtractionError,
                    ValueError,
                    RuntimeError,
                ) as exc:
                    st.session_state.last_error = str(exc)
                else:
                    st.rerun()

        if st.session_state.last_error:
            render_failure_panel(
                "The evidence could not be validated",
                str(st.session_state.last_error),
            )


def _rejection_table(result: object) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Record": rejection.record_index + 1,
                "Reason": rejection.message,
                "Code": rejection.code,
                "Suggested action": REJECTION_ACTIONS.get(
                    rejection.code, "Correct the record and validate again."
                ),
            }
            for rejection in result.rejected_transactions
        ]
    )


def _render_financial_preview(feature: pd.Series) -> None:
    first, second = st.columns(2, gap="large")
    with first:
        render_panel_heading("Cash-flow behaviour")
        render_metric_rows(
            (
                (
                    "Observed inflows",
                    f"{_format_number(feature['inflow_total'])} units",
                ),
                (
                    "Observed outflows",
                    f"{_format_number(feature['outflow_total'])} units",
                ),
                ("Net flow", f"{_format_number(feature['net_flow_total'])} units"),
                (
                    "Negative-flow months",
                    _format_number(feature["negative_net_months"], decimals=0),
                ),
            )
        )
    with second:
        render_panel_heading("Account activity")
        render_metric_rows(
            (
                ("Transactions / month", _format_number(feature["tx_per_month"])),
                ("Active months", _format_number(feature["active_months"], decimals=0)),
                ("Mean balance", f"{_format_number(feature['mean_balance'])} units"),
                ("Low-balance rate", f"{float(feature['low_balance_rate']):.1%}"),
            )
        )

    with st.expander("Review all model input features"):
        names = [name for name in feature.index if name != "applicant_id"]
        st.dataframe(
            pd.DataFrame(
                {"Feature": names, "Value": [feature[name] for name in names]}
            ),
            hide_index=True,
            use_container_width=True,
        )


def _render_validation_step() -> None:
    result = st.session_state.normalization_result
    if result is None:
        st.session_state.assessment_step = 1
        st.rerun()
        return

    with st.container(border=True):
        render_panel_heading(
            "Validation report", SOURCE_OPTIONS[st.session_state.source_key]
        )
        render_validation_counters(
            result.processed_count,
            result.valid_count,
            result.rejected_count,
            len(result.warnings),
        )
        if result.rejected_count:
            st.markdown("#### Records needing correction")
            st.dataframe(
                _rejection_table(result), hide_index=True, use_container_width=True
            )
        if result.warnings:
            with st.expander(f"Review {len(result.warnings)} normalization warnings"):
                for warning in result.warnings:
                    st.write(f"Record {warning.record_index + 1}: {warning.message}")

        if not result.valid_count:
            render_failure_panel(
                "No valid transactions are available",
                "Correct the rejected records and validate the evidence again. "
                "An assessment cannot run without accepted transaction evidence.",
            )
        else:
            st.markdown("#### Financial summary")
            st.caption(
                "Amounts remain in provider wallet units. No currency conversion has "
                "been inferred."
            )
            _render_financial_preview(st.session_state.feature_preview)

        previous, proceed = st.columns([1, 3])
        with previous:
            if st.button("Back to evidence", use_container_width=True):
                st.session_state.assessment_step = 1
                st.rerun()
        with proceed:
            if st.button(
                "Continue to assessment",
                type="primary",
                use_container_width=True,
                disabled=not result.valid_count,
            ):
                st.session_state.assessment_step = 3
                st.rerun()


def _render_assessment_placeholder() -> None:
    with st.container(border=True):
        render_panel_heading(
            "Run assessment", f"Member {st.session_state.applicant_id}"
        )
        st.info(
            "Validated features are ready. Model execution and the explained result "
            "panel are added in the next implementation phase."
        )
        if st.button("Back to validation"):
            st.session_state.assessment_step = 2
            st.rerun()


def render_new_assessment() -> None:
    """Render the current step in the guided assessment workflow."""
    step = int(st.session_state.assessment_step)
    render_page_header(
        "New assessment",
        "Validate transaction evidence before running an explained local credit "
        "assessment.",
        eyebrow=f"Step {min(step + 1, 5)} of 5",
    )
    render_step_bar(step)

    if step == 0:
        _render_applicant_step()
    elif step == 1:
        _render_transaction_step()
    elif step == 2:
        _render_validation_step()
    else:
        _render_assessment_placeholder()
