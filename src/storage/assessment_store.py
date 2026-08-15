"""Transactional SQLite persistence for assessments and human decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import sqlite3

from src.application.assessment import AssessmentResult
from src.storage.db import save_decision, save_explanation, save_features, save_score
from src.xai.narratives import NarrativeFactor
from src.xai.shap_explainer import FeatureContribution


class AssessmentPersistenceError(RuntimeError):
    """Raised when an assessment cannot be persisted atomically."""


class HumanDecision(str, Enum):
    """Controlled values available to an authorized human reviewer."""

    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    DECLINE = "DECLINE"


@dataclass(frozen=True)
class PersistedAssessment:
    """SQLite identifiers written for one atomic assessment operation."""

    feature_id: int
    score_id: int
    explanation_id: int


def _serialize_contribution(contribution: FeatureContribution) -> dict[str, object]:
    return {
        "feature_name": contribution.feature_name,
        "feature_value": contribution.feature_value,
        "shap_value": contribution.shap_value,
        "direction": contribution.direction.value,
        "absolute_importance": contribution.absolute_importance,
    }


def _serialize_narrative_factor(factor: NarrativeFactor) -> dict[str, object]:
    return {
        "feature_name": factor.feature_name,
        "feature_label": factor.feature_label,
        "feature_value": factor.feature_value,
        "shap_value": factor.shap_value,
        "direction": factor.direction.value,
        "text": factor.text,
    }


def persist_assessment(
    connection: sqlite3.Connection,
    assessment: AssessmentResult,
) -> PersistedAssessment:
    """Persist features, score, explanation, and narrative in one transaction."""
    if not isinstance(assessment, AssessmentResult):
        raise TypeError("assessment must be an AssessmentResult.")

    explanation_payload = {
        "risk_score": assessment.explanation.risk_score,
        "base_value": assessment.explanation.base_value,
        "output_space": assessment.explanation.output_space,
        "contributions": [
            _serialize_contribution(contribution)
            for contribution in assessment.explanation.contributions
        ],
        "increasing_risk_features": [
            contribution.feature_name
            for contribution in assessment.explanation.increasing_risk_factors
        ],
        "reducing_risk_features": [
            contribution.feature_name
            for contribution in assessment.explanation.reducing_risk_factors
        ],
    }
    narrative_payload = {
        "summary": assessment.narrative.summary,
        "increasing_risk_factors": [
            _serialize_narrative_factor(factor)
            for factor in assessment.narrative.increasing_risk_factors
        ],
        "reducing_risk_factors": [
            _serialize_narrative_factor(factor)
            for factor in assessment.narrative.reducing_risk_factors
        ],
        "disclaimer": assessment.narrative.disclaimer,
    }

    try:
        with connection:
            feature_id = save_features(
                connection,
                assessment.applicant_id,
                dict(assessment.features),
                commit=False,
            )
            score_id = save_score(
                connection,
                assessment.applicant_id,
                assessment.risk_score,
                assessment.model_version,
                commit=False,
            )
            explanation_id = save_explanation(
                connection,
                assessment.applicant_id,
                assessment.model_version,
                explanation_payload,
                assessment.narrative.language.value,
                narrative_payload,
                commit=False,
            )
    except sqlite3.DatabaseError as exc:
        raise AssessmentPersistenceError(
            f"Could not persist assessment for '{assessment.applicant_id}'."
        ) from exc

    return PersistedAssessment(
        feature_id=feature_id,
        score_id=score_id,
        explanation_id=explanation_id,
    )


def record_human_decision(
    connection: sqlite3.Connection,
    applicant_id: str,
    decision: HumanDecision | str,
    rationale: str | None = None,
) -> int:
    """Validate and persist a human-selected decision independently of scores."""
    if not isinstance(applicant_id, str) or not applicant_id.strip():
        raise ValueError("applicant_id must be a non-empty string.")
    try:
        validated_decision = (
            decision
            if isinstance(decision, HumanDecision)
            else HumanDecision(str(decision).strip().upper())
        )
    except ValueError as exc:
        allowed = ", ".join(item.value for item in HumanDecision)
        raise ValueError(
            f"Unsupported human decision. Allowed values: {allowed}."
        ) from exc

    normalized_rationale = (
        rationale.strip() if rationale and rationale.strip() else None
    )
    return save_decision(
        connection,
        applicant_id.strip(),
        validated_decision.value,
        normalized_rationale,
    )
