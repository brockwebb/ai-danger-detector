from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvidenceType(str, Enum):
    INCIDENT = "incident"
    NEAR_MISS = "near_miss"
    BENIGN_COMPARISON = "benign_comparison"
    BENCHMARK = "benchmark"
    EXPERT_ELICITATION = "expert_elicitation"
    DEPLOYMENT_LOG = "deployment_log"
    USER_STUDY = "user_study"
    CASE_REVIEW = "case_review"
    SYNTHETIC_STRESS_CASE = "synthetic_stress_case"


class OutcomeLabel(str, Enum):
    HARM = "harm"
    LOSS = "loss"
    NEAR_MISS = "near_miss"
    CORRECTED_ERROR = "corrected_error"
    BENIGN_USE = "benign_use"
    UNRESOLVED = "unresolved"
    UNKNOWN = "unknown"


class OversightLabel(str, Enum):
    CASUAL_EXPLORATORY = "casual_exploratory"
    ASSISTED_BOUNDED = "assisted_bounded"
    TRAINED_REVIEW_REQUIRED = "trained_review_required"
    EXPERT_REVIEW_REQUIRED = "expert_review_required"
    EXPERT_LED_OR_NO_AUTONOMOUS_USE = "expert_led_or_no_autonomous_use"
    UNKNOWN = "unknown"


class UserExpertise(str, Enum):
    NON_EXPERT = "non_expert"
    TRAINED = "trained"
    DOMAIN_FAMILIAR = "domain_familiar"
    EXPERT = "expert"
    UNKNOWN = "unknown"


class EvidenceQualityTier(str, Enum):
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4 = "tier_4"
    QUARANTINED = "quarantined"


_UNCERTAINTY_MULTIPLIERS = {
    EvidenceQualityTier.TIER_1: 1.0,
    EvidenceQualityTier.TIER_2: 1.2,
    EvidenceQualityTier.TIER_3: 1.5,
    EvidenceQualityTier.TIER_4: 2.0,
    EvidenceQualityTier.QUARANTINED: 3.0,
}


def _require_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _probability(value: float, field_name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return value


def _as_tuple(values: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    return tuple(str(value).strip() for value in values if str(value).strip())


@dataclass(frozen=True)
class EvidenceUnit:
    evidence_id: str
    source_id: str
    evidence_type: EvidenceType
    collection_date: str
    event_date: str | None
    domain: str
    task_type: str
    model_family: str
    model_version: str
    user_expertise: UserExpertise
    governance_context: str
    outcome_label: OutcomeLabel
    oversight_label: OversightLabel
    harm_severity: float
    detectability: float
    reversibility: float
    verification_burden: float
    workflow_path: tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 0.5
    source_quality: EvidenceQualityTier = EvidenceQualityTier.TIER_3
    bias_notes: tuple[str, ...] = field(default_factory=tuple)
    relevance_limits: tuple[str, ...] = field(default_factory=tuple)
    optional_fields: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "evidence_id", _require_text(self.evidence_id, "evidence_id")
        )
        object.__setattr__(
            self, "source_id", _require_text(self.source_id, "source_id")
        )
        object.__setattr__(
            self,
            "collection_date",
            _require_text(self.collection_date, "collection_date"),
        )
        object.__setattr__(self, "domain", _require_text(self.domain, "domain"))
        object.__setattr__(
            self, "task_type", _require_text(self.task_type, "task_type")
        )
        object.__setattr__(
            self, "model_family", _require_text(self.model_family, "model_family")
        )
        object.__setattr__(
            self, "model_version", _require_text(self.model_version, "model_version")
        )
        object.__setattr__(
            self,
            "governance_context",
            _require_text(self.governance_context, "governance_context"),
        )
        object.__setattr__(
            self, "harm_severity", _probability(self.harm_severity, "harm_severity")
        )
        object.__setattr__(
            self, "detectability", _probability(self.detectability, "detectability")
        )
        object.__setattr__(
            self, "reversibility", _probability(self.reversibility, "reversibility")
        )
        object.__setattr__(
            self,
            "verification_burden",
            _probability(self.verification_burden, "verification_burden"),
        )
        object.__setattr__(
            self, "confidence", _probability(self.confidence, "confidence")
        )
        object.__setattr__(self, "workflow_path", _as_tuple(self.workflow_path))
        object.__setattr__(self, "bias_notes", _as_tuple(self.bias_notes))
        object.__setattr__(
            self, "relevance_limits", _as_tuple(self.relevance_limits)
        )

    @property
    def is_calibration_eligible(self) -> bool:
        return self.source_quality is not EvidenceQualityTier.QUARANTINED

    @property
    def uncertainty_multiplier(self) -> float:
        return _UNCERTAINTY_MULTIPLIERS[self.source_quality]

    def to_feature_row(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_id": self.source_id,
            "evidence_type": self.evidence_type.value,
            "collection_date": self.collection_date,
            "event_date": self.event_date,
            "domain": self.domain,
            "task_type": self.task_type,
            "model_family": self.model_family,
            "model_version": self.model_version,
            "user_expertise": self.user_expertise.value,
            "governance_context": self.governance_context,
            "outcome_label": self.outcome_label.value,
            "oversight_label": self.oversight_label.value,
            "harm_severity": self.harm_severity,
            "detectability": self.detectability,
            "reversibility": self.reversibility,
            "verification_burden": self.verification_burden,
            "workflow_path": self.workflow_path,
            "confidence": self.confidence,
            "source_quality": self.source_quality.value,
            "bias_notes": self.bias_notes,
            "relevance_limits": self.relevance_limits,
            **self.optional_fields,
        }
