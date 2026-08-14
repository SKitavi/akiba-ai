"""Deterministic English and Kiswahili model-risk narratives.

This module converts structured SHAP contributions into concise explanations
for SACCO loan officers. It uses local templates only: no network, translation,
or generative-AI service is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from src.xai.shap_explainer import (
    ContributionDirection,
    FeatureContribution,
    PredictionExplanation,
)


class NarrativeLanguage(str, Enum):
    """Languages currently supported by the narrative generator."""

    ENGLISH = "en"
    KISWAHILI = "sw"


@dataclass(frozen=True)
class NarrativeFactor:
    """A human-readable rendering of one traceable model contribution."""

    feature_name: str
    feature_label: str
    feature_value: float
    shap_value: float
    direction: ContributionDirection
    text: str


@dataclass(frozen=True)
class RiskNarrative:
    """Localized explanation sections ready for a UI or report."""

    language: NarrativeLanguage
    risk_score: float
    summary: str
    increasing_risk_factors: tuple[NarrativeFactor, ...]
    reducing_risk_factors: tuple[NarrativeFactor, ...]
    disclaimer: str


# Every canonical model feature has a centralized user-facing label. Keeping
# labels separate from rendering makes future Kinyarwanda support additive.
FEATURE_LABELS: dict[str, dict[NarrativeLanguage, str]] = {
    "tx_per_month": {
        NarrativeLanguage.ENGLISH: "Transactions per month",
        NarrativeLanguage.KISWAHILI: "Miamala kwa mwezi",
    },
    "peak_week_tx": {
        NarrativeLanguage.ENGLISH: "Highest weekly transaction activity",
        NarrativeLanguage.KISWAHILI: "Kiwango cha juu cha miamala kwa wiki",
    },
    "active_months": {
        NarrativeLanguage.ENGLISH: "Months with transaction activity",
        NarrativeLanguage.KISWAHILI: "Miezi yenye miamala",
    },
    "days_since_last_inflow": {
        NarrativeLanguage.ENGLISH: "Days since the latest incoming funds",
        NarrativeLanguage.KISWAHILI: "Siku tangu fedha zilipoingia mara ya mwisho",
    },
    "net_flow_total": {
        NarrativeLanguage.ENGLISH: "Total net cash flow",
        NarrativeLanguage.KISWAHILI: "Jumla ya mtiririko halisi wa fedha",
    },
    "net_flow_mean": {
        NarrativeLanguage.ENGLISH: "Average monthly net cash flow",
        NarrativeLanguage.KISWAHILI: "Wastani wa mtiririko halisi wa fedha kwa mwezi",
    },
    "net_flow_std": {
        NarrativeLanguage.ENGLISH: "Variation in monthly net cash flow",
        NarrativeLanguage.KISWAHILI: "Mabadiliko ya mtiririko halisi wa fedha kwa mwezi",
    },
    "net_flow_cv": {
        NarrativeLanguage.ENGLISH: "Relative variability of net cash flow",
        NarrativeLanguage.KISWAHILI: "Utofauti wa mtiririko halisi wa fedha",
    },
    "net_flow_ratio": {
        NarrativeLanguage.ENGLISH: "Net cash flow relative to income",
        NarrativeLanguage.KISWAHILI: "Uwiano wa mtiririko halisi wa fedha na mapato",
    },
    "negative_net_months": {
        NarrativeLanguage.ENGLISH: "Months with negative net cash flow",
        NarrativeLanguage.KISWAHILI: "Miezi yenye mtiririko hasi wa fedha",
    },
    "inflow_total": {
        NarrativeLanguage.ENGLISH: "Total incoming funds",
        NarrativeLanguage.KISWAHILI: "Jumla ya fedha zinazoingia",
    },
    "inflow_mean": {
        NarrativeLanguage.ENGLISH: "Average incoming transaction amount",
        NarrativeLanguage.KISWAHILI: "Wastani wa kiasi cha fedha zinazoingia",
    },
    "inflow_std": {
        NarrativeLanguage.ENGLISH: "Variation in incoming transaction amounts",
        NarrativeLanguage.KISWAHILI: "Mabadiliko ya kiasi cha fedha zinazoingia",
    },
    "inflow_cv": {
        NarrativeLanguage.ENGLISH: "Relative variability of incoming funds",
        NarrativeLanguage.KISWAHILI: "Utofauti wa fedha zinazoingia",
    },
    "inflow_per_month": {
        NarrativeLanguage.ENGLISH: "Incoming transactions per month",
        NarrativeLanguage.KISWAHILI: "Miamala ya fedha zinazoingia kwa mwezi",
    },
    "inflow_regularity": {
        NarrativeLanguage.ENGLISH: "Regularity of incoming funds",
        NarrativeLanguage.KISWAHILI: "Uthabiti wa fedha zinazoingia",
    },
    "outflow_total": {
        NarrativeLanguage.ENGLISH: "Total outgoing funds",
        NarrativeLanguage.KISWAHILI: "Jumla ya fedha zinazotoka",
    },
    "outflow_mean": {
        NarrativeLanguage.ENGLISH: "Average outgoing transaction amount",
        NarrativeLanguage.KISWAHILI: "Wastani wa kiasi cha fedha zinazotoka",
    },
    "outflow_std": {
        NarrativeLanguage.ENGLISH: "Variation in outgoing transaction amounts",
        NarrativeLanguage.KISWAHILI: "Mabadiliko ya kiasi cha fedha zinazotoka",
    },
    "outflow_cv": {
        NarrativeLanguage.ENGLISH: "Relative variability of outgoing funds",
        NarrativeLanguage.KISWAHILI: "Utofauti wa fedha zinazotoka",
    },
    "outflow_per_month": {
        NarrativeLanguage.ENGLISH: "Outgoing transactions per month",
        NarrativeLanguage.KISWAHILI: "Miamala ya fedha zinazotoka kwa mwezi",
    },
    "productive_ratio": {
        NarrativeLanguage.ENGLISH: "Share of payments for goods and services",
        NarrativeLanguage.KISWAHILI: "Uwiano wa malipo ya bidhaa na huduma",
    },
    "inflow_outflow_ratio": {
        NarrativeLanguage.ENGLISH: "Ratio of incoming to outgoing funds",
        NarrativeLanguage.KISWAHILI: "Uwiano wa fedha zinazoingia na zinazotoka",
    },
    "low_balance_events": {
        NarrativeLanguage.ENGLISH: "Low wallet balance events",
        NarrativeLanguage.KISWAHILI: "Matukio ya salio dogo la pochi",
    },
    "low_balance_rate": {
        NarrativeLanguage.ENGLISH: "Frequency of low wallet balances",
        NarrativeLanguage.KISWAHILI: "Marudio ya salio dogo la pochi",
    },
    "min_balance": {
        NarrativeLanguage.ENGLISH: "Lowest wallet balance",
        NarrativeLanguage.KISWAHILI: "Salio la chini kabisa la pochi",
    },
    "mean_balance": {
        NarrativeLanguage.ENGLISH: "Average wallet balance",
        NarrativeLanguage.KISWAHILI: "Wastani wa salio la pochi",
    },
    "balance_trend_slope": {
        NarrativeLanguage.ENGLISH: "Wallet balance trend",
        NarrativeLanguage.KISWAHILI: "Mwelekeo wa salio la pochi",
    },
    "airtime_ratio": {
        NarrativeLanguage.ENGLISH: "Share of airtime purchases",
        NarrativeLanguage.KISWAHILI: "Uwiano wa manunuzi ya muda wa maongezi",
    },
    "cashout_ratio": {
        NarrativeLanguage.ENGLISH: "Share of cash withdrawals",
        NarrativeLanguage.KISWAHILI: "Uwiano wa kutoa fedha taslimu",
    },
    "p2p_send_ratio": {
        NarrativeLanguage.ENGLISH: "Share of peer-to-peer transfers sent",
        NarrativeLanguage.KISWAHILI: "Uwiano wa fedha zilizotumwa kwa watu wengine",
    },
    "p2p_receive_ratio": {
        NarrativeLanguage.ENGLISH: "Share of peer-to-peer transfers received",
        NarrativeLanguage.KISWAHILI: "Uwiano wa fedha zilizopokelewa kutoka kwa watu wengine",
    },
}


_TEXT: dict[NarrativeLanguage, dict[str, str]] = {
    NarrativeLanguage.ENGLISH: {
        "summary": (
            "The model produced a risk score of {risk_score:.3f}. The factors below "
            "show the strongest movements in the model output."
        ),
        "increases_risk": (
            "{feature_label} pushed the model output toward higher estimated risk."
        ),
        "reduces_risk": (
            "{feature_label} pushed the model output toward lower estimated risk."
        ),
        "disclaimer": (
            "This explanation describes model behavior, not causation. This MVP uses "
            "synthetic data, and its output should support rather than replace human review."
        ),
    },
    NarrativeLanguage.KISWAHILI: {
        "summary": (
            "Modeli ilitoa alama ya hatari ya {risk_score:.3f}. Vipengele vifuatavyo "
            "vinaonyesha mabadiliko makubwa zaidi katika matokeo ya modeli."
        ),
        "increases_risk": (
            "Kipengele cha {feature_label} kilisukuma matokeo ya modeli kuelekea "
            "makadirio ya hatari kubwa."
        ),
        "reduces_risk": (
            "Kipengele cha {feature_label} kilisukuma matokeo ya modeli kuelekea "
            "makadirio ya hatari ndogo."
        ),
        "disclaimer": (
            "Maelezo haya yanafafanua tabia ya modeli, si uhusiano wa sababu na matokeo. "
            "MVP hii hutumia data sanisi, na matokeo yake yanapaswa kusaidia badala ya "
            "kuchukua nafasi ya mapitio ya binadamu."
        ),
    },
}


def _parse_language(language: str | NarrativeLanguage) -> NarrativeLanguage:
    if isinstance(language, NarrativeLanguage):
        return language
    if not isinstance(language, str):
        raise TypeError("language must be a string or NarrativeLanguage.")
    try:
        return NarrativeLanguage(language.lower())
    except ValueError as exc:
        supported = ", ".join(item.value for item in NarrativeLanguage)
        raise ValueError(
            f"Unsupported language '{language}'. Supported languages: {supported}."
        ) from exc


def get_feature_label(
    feature_name: str, language: str | NarrativeLanguage = "en"
) -> str:
    """Return a localized feature label with a readable unknown-feature fallback."""
    selected_language = _parse_language(language)
    labels = FEATURE_LABELS.get(feature_name)
    if labels is not None:
        return labels[selected_language]
    return feature_name.replace("_", " ").strip().capitalize() or feature_name


def _render_factor(
    contribution: FeatureContribution,
    language: NarrativeLanguage,
) -> NarrativeFactor:
    if contribution.direction is ContributionDirection.NEUTRAL:
        raise ValueError("Neutral contributions do not have directional narratives.")

    feature_label = get_feature_label(contribution.feature_name, language)
    template_key = contribution.direction.value
    text = _TEXT[language][template_key].format(feature_label=feature_label)
    return NarrativeFactor(
        feature_name=contribution.feature_name,
        feature_label=feature_label,
        feature_value=contribution.feature_value,
        shap_value=contribution.shap_value,
        direction=contribution.direction,
        text=text,
    )


def _render_factors(
    contributions: tuple[FeatureContribution, ...],
    expected_direction: ContributionDirection,
    language: NarrativeLanguage,
) -> tuple[NarrativeFactor, ...]:
    for contribution in contributions:
        if contribution.direction is not expected_direction:
            raise ValueError(
                f"Factor '{contribution.feature_name}' has direction "
                f"'{contribution.direction.value}' but is in the "
                f"'{expected_direction.value}' collection."
            )
    return tuple(_render_factor(factor, language) for factor in contributions)


def generate_risk_narrative(
    explanation: PredictionExplanation,
    language: str | NarrativeLanguage = "en",
) -> RiskNarrative:
    """Convert a structured SHAP explanation into localized narrative sections.

    The input rankings are preserved exactly; the generator neither invents
    missing factors nor applies lending thresholds or decisions.

    Args:
        explanation: Result returned by ``explain_prediction``.
        language: ISO-style language code ``en`` or ``sw``.

    Returns:
        A deterministic, structured narrative suitable for Streamlit or reports.

    Raises:
        TypeError: If the explanation or language has the wrong type.
        ValueError: If the language is unsupported or a ranked factor has an
                    invalid direction.
    """
    if not isinstance(explanation, PredictionExplanation):
        raise TypeError("explanation must be a PredictionExplanation.")

    selected_language = _parse_language(language)
    text = _TEXT[selected_language]
    increasing = _render_factors(
        explanation.increasing_risk_factors,
        ContributionDirection.INCREASES_RISK,
        selected_language,
    )
    reducing = _render_factors(
        explanation.reducing_risk_factors,
        ContributionDirection.REDUCES_RISK,
        selected_language,
    )

    return RiskNarrative(
        language=selected_language,
        risk_score=explanation.risk_score,
        summary=text["summary"].format(risk_score=explanation.risk_score),
        increasing_risk_factors=increasing,
        reducing_risk_factors=reducing,
        disclaimer=text["disclaimer"],
    )


def build_narrative(explanation: Mapping[str, float], language: str = "en") -> str:
    """Render a compact narrative from a legacy feature-to-SHAP mapping.

    This compatibility helper treats mapping values as SHAP contributions,
    orders factors by magnitude and name, omits zeros, and appends the standard
    responsible-AI disclaimer. New code should use ``generate_risk_narrative``
    with the structured ``PredictionExplanation`` result.
    """
    selected_language = _parse_language(language)
    contributions = []
    for feature_name, shap_value in explanation.items():
        numeric_value = float(shap_value)
        if numeric_value == 0.0:
            continue
        direction = (
            ContributionDirection.INCREASES_RISK
            if numeric_value > 0.0
            else ContributionDirection.REDUCES_RISK
        )
        contribution = FeatureContribution(
            feature_name=feature_name,
            feature_value=float("nan"),
            shap_value=numeric_value,
            direction=direction,
        )
        contributions.append(contribution)

    contributions.sort(
        key=lambda factor: (-factor.absolute_importance, factor.feature_name)
    )
    factors = [
        _render_factor(contribution, selected_language).text
        for contribution in contributions
    ]
    factors.append(_TEXT[selected_language]["disclaimer"])
    return " ".join(factors)
