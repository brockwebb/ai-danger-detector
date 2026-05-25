from __future__ import annotations

from .evidence_schema import EvidenceUnit, OversightLabel
from .model_comparison import ScorerDefinition, ScorerOutputType
from .rubric_scorer import RubricScore


_DEFERRED_PROBABILITY_METRICS = (
    "brier_score",
    "log_loss",
    "expected_calibration_error",
)


def _fixed_policy_score(band: OversightLabel, name: str) -> RubricScore:
    return RubricScore(
        score=0.0,
        band=band,
        factor_scores={},
        drivers=(f"fixed policy baseline: {name}",),
        assumptions=(
            "Deterministic TEVV policy baseline, not a learned risk model.",
            "Does not emit calibrated probabilities.",
        ),
    )


def _fixed_policy_scorer(
    *,
    name: str,
    description: str,
    band: OversightLabel,
) -> ScorerDefinition:
    def scorer(unit: EvidenceUnit) -> RubricScore:
        return _fixed_policy_score(band, name)

    return ScorerDefinition(
        name=name,
        description=description,
        output_type=ScorerOutputType.ORDINAL_BAND,
        scorer=scorer,
        native_metric_notes=(
            "TEVV control baseline for ordinal and threshold decision metrics.",
            "Useful for exposing false reassurance and over-escalation tradeoffs.",
            "Brier score, log loss, and expected calibration error are deferred because this policy does not emit calibrated probabilities.",
        ),
        equivalence_notes=(
            "The policy returns a fixed oversight band directly.",
            "This is not learned from evidence and should not be treated as a risk model.",
        ),
        deferred_metric_names=_DEFERRED_PROBABILITY_METRICS,
    )


def always_casual_scorer() -> ScorerDefinition:
    return _fixed_policy_scorer(
        name="always_casual",
        description="Fixed policy baseline that always predicts casual exploratory use.",
        band=OversightLabel.CASUAL_EXPLORATORY,
    )


def always_assisted_scorer() -> ScorerDefinition:
    return _fixed_policy_scorer(
        name="always_assisted",
        description="Fixed policy baseline that always predicts assisted bounded use.",
        band=OversightLabel.ASSISTED_BOUNDED,
    )


def always_trained_review_scorer() -> ScorerDefinition:
    return _fixed_policy_scorer(
        name="always_trained_review",
        description="Fixed policy baseline that always predicts trained review.",
        band=OversightLabel.TRAINED_REVIEW_REQUIRED,
    )


def always_expert_review_scorer() -> ScorerDefinition:
    return _fixed_policy_scorer(
        name="always_expert_review",
        description="Fixed policy baseline that always predicts expert review.",
        band=OversightLabel.EXPERT_REVIEW_REQUIRED,
    )


def always_expert_led_scorer() -> ScorerDefinition:
    return _fixed_policy_scorer(
        name="always_expert_led",
        description="Fixed policy baseline that always predicts expert-led or no autonomous use.",
        band=OversightLabel.EXPERT_LED_OR_NO_AUTONOMOUS_USE,
    )


def reference_policy_scorers() -> tuple[ScorerDefinition, ...]:
    return (
        always_casual_scorer(),
        always_assisted_scorer(),
        always_trained_review_scorer(),
        always_expert_review_scorer(),
        always_expert_led_scorer(),
    )
