from __future__ import annotations

from dataclasses import dataclass

from .evidence_schema import (
    EvidenceQualityTier,
    EvidenceUnit,
    OversightLabel,
    UserExpertise,
)


@dataclass(frozen=True)
class RubricScoringConfig:
    weights: dict[str, float]
    thresholds: tuple[tuple[float, OversightLabel], ...]


@dataclass(frozen=True)
class RubricScore:
    score: float
    band: OversightLabel
    factor_scores: dict[str, float]
    drivers: tuple[str, ...]
    assumptions: tuple[str, ...]


DEFAULT_RUBRIC_CONFIG = RubricScoringConfig(
    weights={
        "harm_severity": 0.24,
        "low_detectability": 0.16,
        "low_reversibility": 0.14,
        "verification_burden": 0.18,
        "user_expertise_risk": 0.12,
        "governance_risk": 0.10,
        "source_uncertainty_risk": 0.06,
    },
    thresholds=(
        (0.80, OversightLabel.EXPERT_LED_OR_NO_AUTONOMOUS_USE),
        (0.60, OversightLabel.EXPERT_REVIEW_REQUIRED),
        (0.40, OversightLabel.TRAINED_REVIEW_REQUIRED),
        (0.20, OversightLabel.ASSISTED_BOUNDED),
        (0.00, OversightLabel.CASUAL_EXPLORATORY),
    ),
)


_USER_EXPERTISE_RISK = {
    UserExpertise.NON_EXPERT: 1.0,
    UserExpertise.UNKNOWN: 0.75,
    UserExpertise.TRAINED: 0.35,
    UserExpertise.DOMAIN_FAMILIAR: 0.25,
    UserExpertise.EXPERT: 0.10,
}

_SOURCE_QUALITY_RISK = {
    EvidenceQualityTier.TIER_1: 0.0,
    EvidenceQualityTier.TIER_2: 0.15,
    EvidenceQualityTier.TIER_3: 0.40,
    EvidenceQualityTier.TIER_4: 0.75,
    EvidenceQualityTier.QUARANTINED: 1.0,
}

_DEFAULT_ASSUMPTIONS = (
    "Score is a provisional rubric baseline, not a calibrated probability.",
    "Governance context is interpreted with a simple text heuristic.",
    "Source quality affects uncertainty rather than proving correctness.",
)


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _risk_from_user_expertise(user_expertise: UserExpertise) -> float:
    return _USER_EXPERTISE_RISK[user_expertise]


def _risk_from_source_quality(source_quality: EvidenceQualityTier) -> float:
    return _SOURCE_QUALITY_RISK[source_quality]


def _risk_from_governance_context(governance_context: str) -> float:
    text = governance_context.lower()
    weak_markers = (
        "no review",
        "no expert",
        "without review",
        "without expert",
        "no clinical",
        "no fiduciary",
        "informal",
    )
    strong_markers = (
        "required expert",
        "expert signoff",
        "formal approval",
        "audit trail",
        "monitoring",
        "code review",
    )
    moderate_markers = (
        "review",
        "approval",
        "second source",
        "teacher reviews",
        "structured",
    )
    if any(marker in text for marker in weak_markers):
        return 0.85
    if any(marker in text for marker in strong_markers):
        return 0.20
    if any(marker in text for marker in moderate_markers):
        return 0.25
    return 0.55


def _band_for_score(score: float, config: RubricScoringConfig) -> OversightLabel:
    for threshold, band in config.thresholds:
        if score >= threshold:
            return band
    return OversightLabel.CASUAL_EXPLORATORY


def _drivers(factor_scores: dict[str, float]) -> tuple[str, ...]:
    labels = {
        "harm_severity": "harm severity",
        "low_detectability": "low detectability",
        "low_reversibility": "low reversibility",
        "verification_burden": "verification burden",
        "user_expertise_risk": "user expertise gap",
        "governance_risk": "weak governance",
        "source_uncertainty_risk": "source uncertainty",
    }
    drivers = [
        labels[name]
        for name, value in factor_scores.items()
        if value >= 0.70
    ]
    return tuple(drivers)


def score_evidence_unit(
    unit: EvidenceUnit,
    *,
    config: RubricScoringConfig = DEFAULT_RUBRIC_CONFIG,
) -> RubricScore:
    factor_scores = {
        "harm_severity": _clamp_probability(unit.harm_severity),
        "low_detectability": 1.0 - _clamp_probability(unit.detectability),
        "low_reversibility": 1.0 - _clamp_probability(unit.reversibility),
        "verification_burden": _clamp_probability(unit.verification_burden),
        "user_expertise_risk": _risk_from_user_expertise(unit.user_expertise),
        "governance_risk": _risk_from_governance_context(unit.governance_context),
        "source_uncertainty_risk": _risk_from_source_quality(unit.source_quality),
    }
    score = sum(
        factor_scores[name] * weight
        for name, weight in config.weights.items()
    )
    score = _clamp_probability(score)
    return RubricScore(
        score=score,
        band=_band_for_score(score, config),
        factor_scores=factor_scores,
        drivers=_drivers(factor_scores),
        assumptions=_DEFAULT_ASSUMPTIONS,
    )
