import pytest

from core_model.evidence_corpus import EvidenceCorpus
from core_model.evidence_schema import (
    EvidenceQualityTier,
    EvidenceType,
    EvidenceUnit,
    OutcomeLabel,
    OversightLabel,
    UserExpertise,
)
from core_model.source_registry import SourceRegistry, SourceStatus, SourceType


def _registry():
    registry = SourceRegistry()
    registry.add_source(
        source_id="src-active",
        source_name="Active adjudicated set",
        source_type=SourceType.CASE_SET,
        owner_or_publisher="ADD",
        license_or_access="private",
        update_cadence="one-time",
        coverage=("health",),
        known_biases=(),
        quality_tier=EvidenceQualityTier.TIER_1,
    )
    registry.update_status("src-active", SourceStatus.ACTIVE, reason="approved")
    return registry


def _evidence(evidence_id="case-001", source_id="src-active", **overrides):
    values = {
        "evidence_id": evidence_id,
        "source_id": source_id,
        "evidence_type": EvidenceType.CASE_REVIEW,
        "collection_date": "2026-05-24",
        "event_date": "2026-05-20",
        "domain": "health",
        "task_type": "medical symptom advice",
        "model_family": "unknown",
        "model_version": "unknown",
        "user_expertise": UserExpertise.NON_EXPERT,
        "governance_context": "informal",
        "outcome_label": OutcomeLabel.HARM,
        "oversight_label": OversightLabel.EXPERT_REVIEW_REQUIRED,
        "harm_severity": 0.8,
        "detectability": 0.2,
        "reversibility": 0.3,
        "verification_burden": 0.9,
        "workflow_path": ("S0", "S1", "S2", "S6", "S8"),
        "confidence": 0.7,
        "source_quality": EvidenceQualityTier.TIER_1,
        "bias_notes": (),
        "relevance_limits": (),
    }
    values.update(overrides)
    return EvidenceUnit(**values)


def test_corpus_adds_evidence_for_known_source():
    corpus = EvidenceCorpus(_registry())
    unit = _evidence()

    corpus.add(unit)

    assert corpus.get("case-001") == unit
    assert corpus.evidence_ids == ("case-001",)


def test_corpus_rejects_unknown_source_id():
    corpus = EvidenceCorpus(_registry())

    with pytest.raises(KeyError, match="unknown source_id"):
        corpus.add(_evidence(source_id="missing-source"))


def test_corpus_rejects_duplicate_evidence_id():
    corpus = EvidenceCorpus(_registry())
    corpus.add(_evidence())

    with pytest.raises(ValueError, match="already exists"):
        corpus.add(_evidence())
