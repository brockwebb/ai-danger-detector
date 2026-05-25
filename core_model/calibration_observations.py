"""Build traceable calibration observations from ADD evidence records."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from .bayesian_calibration import BetaObservation
from .evidence_corpus import EvidenceCorpus
from .evidence_schema import EvidenceQualityTier, EvidenceUnit, OutcomeLabel
from .source_registry import SourceStatus


class CalibrationParameter(str, Enum):
    P_ERROR_PER_TASK = "p_error_per_task"
    DETECTABILITY = "detectability"
    REVERSIBILITY = "reversibility"


DEFAULT_QUALITY_WEIGHTS = {
    EvidenceQualityTier.TIER_1: 1.0,
    EvidenceQualityTier.TIER_2: 0.75,
    EvidenceQualityTier.TIER_3: 0.5,
    EvidenceQualityTier.TIER_4: 0.25,
    EvidenceQualityTier.QUARANTINED: 0.0,
}

_EXCLUDED_SOURCE_STATUSES = {
    SourceStatus.QUARANTINED,
    SourceStatus.DEPRECATED,
    SourceStatus.REMOVED,
}

_EVENT_OUTCOMES = {
    OutcomeLabel.HARM,
    OutcomeLabel.LOSS,
    OutcomeLabel.NEAR_MISS,
    OutcomeLabel.CORRECTED_ERROR,
}


@dataclass(frozen=True)
class CalibrationObservationConfig:
    include_experimental_sources: bool = False
    config_version: str = "calibration-observation-v1"
    pseudo_observation_strength: float = 1.0
    quality_weights: dict[EvidenceQualityTier, float] = field(
        default_factory=lambda: dict(DEFAULT_QUALITY_WEIGHTS)
    )
    minimum_confidence: float = 0.0

    def __post_init__(self) -> None:
        config_version = str(self.config_version).strip()
        if not config_version:
            raise ValueError("config_version must be non-empty")

        pseudo_observation_strength = float(self.pseudo_observation_strength)
        if pseudo_observation_strength <= 0:
            raise ValueError("pseudo_observation_strength must be positive")

        minimum_confidence = float(self.minimum_confidence)
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")

        quality_weights = dict(DEFAULT_QUALITY_WEIGHTS)
        for tier, weight in self.quality_weights.items():
            quality_tier = EvidenceQualityTier(tier)
            weight = float(weight)
            if weight < 0:
                raise ValueError("quality weight must be non-negative")
            quality_weights[quality_tier] = weight

        object.__setattr__(
            self, "include_experimental_sources", bool(self.include_experimental_sources)
        )
        object.__setattr__(self, "config_version", config_version)
        object.__setattr__(
            self, "pseudo_observation_strength", pseudo_observation_strength
        )
        object.__setattr__(self, "quality_weights", quality_weights)
        object.__setattr__(self, "minimum_confidence", minimum_confidence)

    def summary(self) -> dict:
        return {
            "include_experimental_sources": self.include_experimental_sources,
            "config_version": self.config_version,
            "pseudo_observation_strength": self.pseudo_observation_strength,
            "quality_weights": {
                tier.value: weight
                for tier, weight in sorted(
                    self.quality_weights.items(), key=lambda item: item[0].value
                )
            },
            "minimum_confidence": self.minimum_confidence,
        }


@dataclass(frozen=True)
class ExcludedEvidence:
    evidence_id: str
    source_id: str
    parameter: CalibrationParameter
    reason_code: str
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CalibrationObservationSet:
    parameter: CalibrationParameter
    observations: tuple[BetaObservation, ...]
    exclusions: tuple[ExcludedEvidence, ...]
    config_version: str

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    observation.source_id
                    for observation in self.observations
                    if observation.source_id is not None
                }
            )
        )

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                evidence_id
                for observation in self.observations
                for evidence_id in observation.evidence_ids
            )
        )

    def summary(self) -> dict:
        exclusion_reasons = Counter(
            exclusion.reason_code for exclusion in self.exclusions
        )
        return {
            "parameter": self.parameter.value,
            "config_version": self.config_version,
            "observation_count": len(self.observations),
            "exclusion_count": len(self.exclusions),
            "source_ids": self.source_ids,
            "evidence_ids": self.evidence_ids,
            "excluded_evidence_ids": tuple(
                exclusion.evidence_id for exclusion in self.exclusions
            ),
            "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        }


def _coerce_parameter(parameter: CalibrationParameter | str) -> CalibrationParameter:
    if isinstance(parameter, CalibrationParameter):
        return parameter
    try:
        return CalibrationParameter(str(parameter))
    except ValueError as exc:
        raise ValueError(f"unsupported calibration parameter: {parameter}") from exc


def _exclude(
    unit: EvidenceUnit,
    parameter: CalibrationParameter,
    reason_code: str,
    notes: tuple[str, ...] = (),
) -> ExcludedEvidence:
    return ExcludedEvidence(
        evidence_id=unit.evidence_id,
        source_id=unit.source_id,
        parameter=parameter,
        reason_code=reason_code,
        notes=notes,
    )


def _source_exclusion(
    corpus: EvidenceCorpus,
    unit: EvidenceUnit,
    parameter: CalibrationParameter,
    config: CalibrationObservationConfig,
) -> ExcludedEvidence | None:
    source = corpus.source_registry.get_source(unit.source_id)
    if source.status is SourceStatus.EXPERIMENTAL and not config.include_experimental_sources:
        return _exclude(
            unit,
            parameter,
            "source_status_experimental",
            (
                "source_status=experimental",
                f"include_experimental_sources={config.include_experimental_sources}",
            ),
        )
    if source.status in _EXCLUDED_SOURCE_STATUSES:
        return _exclude(
            unit,
            parameter,
            f"source_status_{source.status.value}",
            (f"source_status={source.status.value}",),
        )
    if not unit.is_calibration_eligible:
        return _exclude(
            unit,
            parameter,
            "record_not_calibration_eligible",
            (f"source_quality={unit.source_quality.value}",),
        )
    return None


def _observation_weight(
    unit: EvidenceUnit, config: CalibrationObservationConfig
) -> float:
    return config.quality_weights[unit.source_quality] * unit.confidence


def _weight_exclusion(
    unit: EvidenceUnit,
    parameter: CalibrationParameter,
    config: CalibrationObservationConfig,
) -> ExcludedEvidence | None:
    if unit.confidence < config.minimum_confidence:
        return _exclude(
            unit,
            parameter,
            "below_minimum_confidence",
            (
                f"confidence={unit.confidence:.3f}",
                f"minimum_confidence={config.minimum_confidence:.3f}",
            ),
        )
    if _observation_weight(unit, config) == 0:
        return _exclude(
            unit,
            parameter,
            "zero_observation_weight",
            (
                f"source_quality={unit.source_quality.value}",
                f"confidence={unit.confidence:.3f}",
            ),
        )
    return None


def _common_notes(
    unit: EvidenceUnit,
    parameter: CalibrationParameter,
    config: CalibrationObservationConfig,
) -> tuple[str, ...]:
    return (
        f"parameter={parameter.value}",
        f"quality_tier={unit.source_quality.value}",
        f"confidence={unit.confidence:.3f}",
        f"config_version={config.config_version}",
    )


def _error_observation(
    unit: EvidenceUnit,
    parameter: CalibrationParameter,
    config: CalibrationObservationConfig,
) -> BetaObservation | ExcludedEvidence:
    if unit.outcome_label in _EVENT_OUTCOMES:
        successes = 1.0
        failures = 0.0
        mapping_note = "mapping=meaningful_error_event"
    elif unit.outcome_label is OutcomeLabel.BENIGN_USE:
        successes = 0.0
        failures = 1.0
        mapping_note = "mapping=benign_non_event"
    else:
        return _exclude(
            unit,
            parameter,
            "unsupported_outcome_label",
            (f"outcome_label={unit.outcome_label.value}",),
        )

    return BetaObservation(
        successes=successes,
        failures=failures,
        weight=_observation_weight(unit, config),
        source_id=unit.source_id,
        evidence_ids=(unit.evidence_id,),
        notes=(
            *_common_notes(unit, parameter, config),
            f"outcome_label={unit.outcome_label.value}",
            mapping_note,
        ),
    )


def _pseudo_observation(
    unit: EvidenceUnit,
    parameter: CalibrationParameter,
    value: float,
    config: CalibrationObservationConfig,
) -> BetaObservation:
    successes = value * config.pseudo_observation_strength
    failures = (1.0 - value) * config.pseudo_observation_strength
    return BetaObservation(
        successes=successes,
        failures=failures,
        weight=_observation_weight(unit, config),
        source_id=unit.source_id,
        evidence_ids=(unit.evidence_id,),
        notes=(
            *_common_notes(unit, parameter, config),
            "pseudo-observation",
            "not empirical event count",
            f"scalar_value={value:.3f}",
            f"pseudo_observation_strength={config.pseudo_observation_strength:.3f}",
        ),
    )


def _map_observation(
    unit: EvidenceUnit,
    parameter: CalibrationParameter,
    config: CalibrationObservationConfig,
) -> BetaObservation | ExcludedEvidence:
    if parameter is CalibrationParameter.P_ERROR_PER_TASK:
        return _error_observation(unit, parameter, config)
    if parameter is CalibrationParameter.DETECTABILITY:
        return _pseudo_observation(unit, parameter, unit.detectability, config)
    if parameter is CalibrationParameter.REVERSIBILITY:
        return _pseudo_observation(unit, parameter, unit.reversibility, config)
    raise ValueError(f"unsupported calibration parameter: {parameter}")


def build_calibration_observations(
    corpus: EvidenceCorpus,
    parameter: CalibrationParameter | str,
    *,
    config: CalibrationObservationConfig | None = None,
) -> CalibrationObservationSet:
    parameter = _coerce_parameter(parameter)
    config = config or CalibrationObservationConfig()
    observations: list[BetaObservation] = []
    exclusions: list[ExcludedEvidence] = []

    for unit in corpus.evidence:
        exclusion = _source_exclusion(corpus, unit, parameter, config)
        if exclusion is not None:
            exclusions.append(exclusion)
            continue

        exclusion = _weight_exclusion(unit, parameter, config)
        if exclusion is not None:
            exclusions.append(exclusion)
            continue

        mapped = _map_observation(unit, parameter, config)
        if isinstance(mapped, ExcludedEvidence):
            exclusions.append(mapped)
        else:
            observations.append(mapped)

    return CalibrationObservationSet(
        parameter=parameter,
        observations=tuple(observations),
        exclusions=tuple(exclusions),
        config_version=config.config_version,
    )
