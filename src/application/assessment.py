"""Application orchestration for one applicant credit-risk assessment."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

import pandas as pd

from src.domain.transactions import NormalizedTransaction, transactions_to_dataframe
from src.features.build_features import FEATURE_COLUMNS, build_feature_table
from src.model.loader import ModelBundle
from src.xai.narratives import (
    NarrativeLanguage,
    RiskNarrative,
    generate_risk_narrative,
)
from src.xai.shap_explainer import PredictionExplanation, explain_prediction


class AssessmentError(RuntimeError):
    """Base error for predictable application-assessment failures."""


class AssessmentInputError(AssessmentError):
    """Raised when the requested applicant assessment scope is invalid."""


class NoTransactionsError(AssessmentInputError):
    """Raised when an assessment has no canonical transactions."""


class ApplicantScopeError(AssessmentInputError):
    """Raised when transactions do not identify exactly the requested applicant."""


class FeatureEngineeringError(AssessmentError):
    """Raised when canonical transactions cannot produce a feature row."""


class AssessmentExecutionError(AssessmentError):
    """Raised when model scoring or SHAP explanation fails."""


@dataclass(frozen=True)
class AssessmentResult:
    """Complete model assessment output for application and persistence layers."""

    applicant_id: str
    model_version: str
    risk_score: float
    features: Mapping[str, float]
    explanation: PredictionExplanation
    narrative: RiskNarrative


def _resolve_applicant_id(
    transactions: tuple[NormalizedTransaction, ...], applicant_id: str | None
) -> str:
    transaction_applicants = {transaction.applicant_id for transaction in transactions}
    if applicant_id is None:
        if len(transaction_applicants) != 1:
            raise ApplicantScopeError(
                "Assessment transactions must belong to exactly one applicant."
            )
        return next(iter(transaction_applicants))

    requested_applicant = applicant_id.strip()
    if not requested_applicant:
        raise ApplicantScopeError("applicant_id must be a non-empty string.")
    if transaction_applicants != {requested_applicant}:
        raise ApplicantScopeError(
            "All transactions must belong to the requested applicant "
            f"'{requested_applicant}'. Found: {sorted(transaction_applicants)}."
        )
    return requested_applicant


def assess_applicant(
    transactions: Iterable[NormalizedTransaction],
    model_bundle: ModelBundle,
    applicant_id: str | None = None,
    applicants_df: pd.DataFrame | None = None,
    language: str | NarrativeLanguage = NarrativeLanguage.ENGLISH,
    top_n: int = 5,
) -> AssessmentResult:
    """Build features, score, explain, and narrate one applicant assessment.

    The operation deliberately returns no approve/review/decline field. Lending
    decisions remain a separate human action and persistence boundary.

    Raises:
        NoTransactionsError: If no canonical transactions are supplied.
        ApplicantScopeError: If transactions span or mismatch applicants.
        FeatureEngineeringError: If feature aggregation fails.
        AssessmentExecutionError: If scoring or SHAP explanation fails.
        AssessmentInputError: If language or ranking options are invalid.
    """
    canonical_transactions = tuple(transactions)
    if not canonical_transactions:
        raise NoTransactionsError("At least one valid transaction is required.")
    if not all(
        isinstance(transaction, NormalizedTransaction)
        for transaction in canonical_transactions
    ):
        raise AssessmentInputError(
            "transactions must contain only NormalizedTransaction records."
        )
    if not isinstance(model_bundle, ModelBundle):
        raise AssessmentInputError("model_bundle must be a validated ModelBundle.")

    resolved_applicant_id = _resolve_applicant_id(canonical_transactions, applicant_id)
    transaction_frame = transactions_to_dataframe(canonical_transactions)

    if applicants_df is not None:
        if "applicant_id" not in applicants_df.columns:
            raise AssessmentInputError(
                "applicants_df must contain an applicant_id column."
            )
        if resolved_applicant_id not in set(applicants_df["applicant_id"]):
            raise AssessmentInputError(
                f"Applicant '{resolved_applicant_id}' is missing from applicants_df."
            )

    try:
        feature_table = build_feature_table(
            transaction_frame, applicants_df=applicants_df
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FeatureEngineeringError(
            f"Could not build features for applicant '{resolved_applicant_id}'."
        ) from exc

    applicant_features = feature_table.loc[
        feature_table["applicant_id"] == resolved_applicant_id
    ]
    if len(applicant_features) != 1:
        raise FeatureEngineeringError(
            f"Expected one feature row for applicant '{resolved_applicant_id}', "
            f"found {len(applicant_features)}."
        )

    try:
        explanation = explain_prediction(
            model=model_bundle.model,
            features=applicant_features,
            top_n=top_n,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise AssessmentExecutionError(
            f"Could not score and explain applicant '{resolved_applicant_id}'."
        ) from exc

    try:
        narrative = generate_risk_narrative(explanation, language=language)
    except (TypeError, ValueError) as exc:
        raise AssessmentInputError(f"Could not generate narrative: {exc}") from exc

    feature_payload = MappingProxyType(
        {
            feature_name: float(applicant_features.iloc[0][feature_name])
            for feature_name in FEATURE_COLUMNS
        }
    )
    return AssessmentResult(
        applicant_id=resolved_applicant_id,
        model_version=model_bundle.model_version,
        risk_score=explanation.risk_score,
        features=feature_payload,
        explanation=explanation,
        narrative=narrative,
    )
