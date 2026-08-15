"""Transactional SQLite persistence for assessments and human decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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
    assessment_id: int


@dataclass(frozen=True)
class AssessmentAuditContext:
    """Optional ingestion-quality metadata attached to one assessment run."""

    source_key: str | None = None
    processed_count: int | None = None
    valid_count: int | None = None
    rejected_count: int | None = None
    warning_count: int | None = None

    def normalized(self) -> "AssessmentAuditContext":
        """Validate counts and normalize an optional source identifier."""
        counts = (
            self.processed_count,
            self.valid_count,
            self.rejected_count,
            self.warning_count,
        )
        if any(
            value is not None and (not isinstance(value, int) or value < 0)
            for value in counts
        ):
            raise ValueError(
                "Audit counts must be non-negative integers when provided."
            )
        transaction_counts = counts[:3]
        if any(value is not None for value in transaction_counts):
            if any(value is None for value in transaction_counts):
                raise ValueError(
                    "processed_count, valid_count, and rejected_count must be "
                    "provided together."
                )
            if self.valid_count + self.rejected_count != self.processed_count:
                raise ValueError(
                    "valid_count plus rejected_count must equal processed_count."
                )
        source_key = self.source_key.strip() if self.source_key else None
        return AssessmentAuditContext(
            source_key=source_key,
            processed_count=self.processed_count,
            valid_count=self.valid_count,
            rejected_count=self.rejected_count,
            warning_count=self.warning_count,
        )


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
    audit_context: AssessmentAuditContext | None = None,
) -> PersistedAssessment:
    """Persist features, score, explanation, and narrative in one transaction."""
    if not isinstance(assessment, AssessmentResult):
        raise TypeError("assessment must be an AssessmentResult.")
    audit = (audit_context or AssessmentAuditContext()).normalized()

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
            created_at = datetime.now(timezone.utc).isoformat()
            cursor = connection.execute(
                """
                INSERT INTO assessment_runs (
                    applicant_id, feature_id, score_id, explanation_id,
                    source_key, processed_count, valid_count, rejected_count,
                    warning_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    assessment.applicant_id,
                    feature_id,
                    score_id,
                    explanation_id,
                    audit.source_key,
                    audit.processed_count,
                    audit.valid_count,
                    audit.rejected_count,
                    audit.warning_count,
                    created_at,
                ),
            )
            assessment_id = int(cursor.lastrowid)
    except sqlite3.DatabaseError as exc:
        raise AssessmentPersistenceError(
            f"Could not persist assessment for '{assessment.applicant_id}'."
        ) from exc

    return PersistedAssessment(
        feature_id=feature_id,
        score_id=score_id,
        explanation_id=explanation_id,
        assessment_id=assessment_id,
    )


def record_human_decision(
    connection: sqlite3.Connection,
    applicant_id: str,
    decision: HumanDecision | str,
    rationale: str | None = None,
    *,
    assessment_id: int | None = None,
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
    normalized_applicant_id = applicant_id.strip()
    if assessment_id is None:
        return save_decision(
            connection,
            normalized_applicant_id,
            validated_decision.value,
            normalized_rationale,
        )
    if not isinstance(assessment_id, int) or assessment_id <= 0:
        raise ValueError("assessment_id must be a positive integer.")

    assessment_row = connection.execute(
        "SELECT applicant_id FROM assessment_runs WHERE assessment_id = ?",
        (assessment_id,),
    ).fetchone()
    if assessment_row is None:
        raise ValueError(f"Assessment {assessment_id} does not exist.")
    if assessment_row[0] != normalized_applicant_id:
        raise ValueError("The assessment does not belong to this applicant.")
    if connection.execute(
        "SELECT 1 FROM assessment_decision_links WHERE assessment_id = ?",
        (assessment_id,),
    ).fetchone():
        raise ValueError("This assessment already has an officer decision.")

    with connection:
        decision_id = save_decision(
            connection,
            normalized_applicant_id,
            validated_decision.value,
            normalized_rationale,
            commit=False,
        )
        connection.execute(
            """
            INSERT INTO assessment_decision_links (assessment_id, decision_id)
            VALUES (?, ?)
            """,
            (assessment_id, decision_id),
        )
    return decision_id
