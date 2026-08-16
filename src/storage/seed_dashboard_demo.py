"""Create deterministic synthetic assessment records for dashboard demonstrations."""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import sqlite3

from src.application.assessment import AssessmentResult
from src.features.build_features import FEATURE_COLUMNS
from src.storage.assessment_store import (
    AssessmentAuditContext,
    HumanDecision,
    persist_assessment,
    record_human_decision,
)
from src.storage.db import get_connection, initialize_schema
from src.xai.narratives import generate_risk_narrative
from src.xai.shap_explainer import (
    ContributionDirection,
    FeatureContribution,
    PredictionExplanation,
)


_SCHEMA_PATH = Path(__file__).with_name("schema.sql")
_SOURCE_KEY = "dashboard_demo"
_MODEL_VERSION = "dashboard_demo_v1"
_SCORES = (0.12, 0.22, 0.31, 0.44, 0.53, 0.68, 0.76, 0.89, 0.18, 0.39, 0.62, 0.92)
_DECISIONS: tuple[HumanDecision | None, ...] = (
    HumanDecision.APPROVE,
    HumanDecision.REVIEW,
    HumanDecision.DECLINE,
    None,
)


@dataclass(frozen=True)
class DashboardSeedResult:
    """Summary of one idempotent dashboard seed operation."""

    requested_count: int
    existing_count: int
    created_count: int


def _build_demo_assessment(index: int) -> AssessmentResult:
    risk_score = _SCORES[(index - 1) % len(_SCORES)]
    features = {feature: 0.0 for feature in FEATURE_COLUMNS}
    features["low_balance_rate"] = risk_score
    features["inflow_regularity"] = 1.0 - risk_score

    increasing = FeatureContribution(
        feature_name="low_balance_rate",
        feature_value=features["low_balance_rate"],
        shap_value=0.25 + (risk_score * 0.2),
        direction=ContributionDirection.INCREASES_RISK,
    )
    reducing = FeatureContribution(
        feature_name="inflow_regularity",
        feature_value=features["inflow_regularity"],
        shap_value=-(0.15 + ((1.0 - risk_score) * 0.2)),
        direction=ContributionDirection.REDUCES_RISK,
    )
    explanation = PredictionExplanation(
        risk_score=risk_score,
        base_value=0.0,
        output_space="raw_margin_log_odds",
        contributions=(increasing, reducing),
        increasing_risk_factors=(increasing,),
        reducing_risk_factors=(reducing,),
    )
    return AssessmentResult(
        applicant_id=f"DASH_DEMO_{index:03d}",
        model_version=_MODEL_VERSION,
        risk_score=risk_score,
        features=features,
        explanation=explanation,
        narrative=generate_risk_narrative(explanation, language="en"),
    )


def seed_dashboard_assessments(
    connection: sqlite3.Connection,
    *,
    count: int = 12,
) -> DashboardSeedResult:
    """Ensure a target number of clearly labelled dashboard demo assessments."""
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 100:
        raise ValueError("count must be an integer between 1 and 100.")

    existing_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM assessment_runs WHERE source_key = ?",
            (_SOURCE_KEY,),
        ).fetchone()[0]
    )
    created_count = max(0, count - existing_count)
    for index in range(existing_count + 1, count + 1):
        assessment = _build_demo_assessment(index)
        processed_count = 24 + index
        rejected_count = 1 if index % 5 == 0 else 0
        stored = persist_assessment(
            connection,
            assessment,
            AssessmentAuditContext(
                source_key=_SOURCE_KEY,
                processed_count=processed_count,
                valid_count=processed_count - rejected_count,
                rejected_count=rejected_count,
                warning_count=1 if index % 4 == 0 else 0,
            ),
        )
        decision = _DECISIONS[(index - 1) % len(_DECISIONS)]
        if decision is not None:
            record_human_decision(
                connection,
                assessment.applicant_id,
                decision,
                rationale="Synthetic dashboard demonstration decision.",
                assessment_id=stored.assessment_id,
            )

    return DashboardSeedResult(
        requested_count=count,
        existing_count=existing_count,
        created_count=created_count,
    )


def main() -> None:
    """Seed the configured SQLite database from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--db-path", type=Path, default=None)
    args = parser.parse_args()
    with closing(get_connection(args.db_path)) as connection:
        initialize_schema(connection, _SCHEMA_PATH)
        result = seed_dashboard_assessments(connection, count=args.count)
    print(
        f"Dashboard demo assessments: {result.created_count} created, "
        f"{result.existing_count} already present, "
        f"{result.requested_count} requested."
    )


if __name__ == "__main__":
    main()
