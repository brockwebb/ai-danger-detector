import pytest

from core_model.evidence_schema import (
    EvidenceQualityTier,
    EvidenceType,
    EvidenceUnit,
    OutcomeLabel,
    OversightLabel,
    UserExpertise,
)


def _valid_evidence(**overrides):
    values = {
        "evidence_id": "case-001",
        "source_id": "src-001",
        "evidence_type": EvidenceType.INCIDENT,
        "collection_date": "2026-05-24",
        "event_date": "2026-05-20",
        "domain": "health",
        "task_type": "medical symptom advice",
        "model_family": "unknown",
        "model_version": "unknown",
        "user_expertise": UserExpertise.NON_EXPERT,
        "governance_context": "informal consumer use",
        "outcome_label": OutcomeLabel.HARM,
        "oversight_label": OversightLabel.EXPERT_REVIEW_REQUIRED,
        "harm_severity": 0.8,
        "detectability": 0.2,
        "reversibility": 0.3,
        "verification_burden": 0.9,
        "workflow_path": ("S0", "S1", "S2", "S6", "S8"),
        "confidence": 0.7,
        "source_quality": EvidenceQualityTier.TIER_3,
        "bias_notes": ("single public report",),
        "relevance_limits": ("consumer health only",),
    }
    values.update(overrides)
    return EvidenceUnit(**values)


def test_evidence_unit_accepts_valid_required_fields():
    unit = _valid_evidence()

    assert unit.evidence_id == "case-001"
    assert unit.is_calibration_eligible is True
    assert unit.uncertainty_multiplier == pytest.approx(1.5)


def test_evidence_unit_rejects_probability_outside_zero_one():
    with pytest.raises(ValueError, match="harm_severity"):
        _valid_evidence(harm_severity=1.5)


def test_evidence_unit_rejects_empty_identifiers():
    with pytest.raises(ValueError, match="evidence_id"):
        _valid_evidence(evidence_id="")


def test_quarantined_evidence_is_not_calibration_eligible():
    unit = _valid_evidence(source_quality=EvidenceQualityTier.QUARANTINED)

    assert unit.is_calibration_eligible is False
    assert unit.uncertainty_multiplier == pytest.approx(3.0)


def test_evidence_unit_serializes_to_model_ready_dict():
    unit = _valid_evidence()

    payload = unit.to_feature_row()

    assert payload["evidence_id"] == "case-001"
    assert payload["evidence_type"] == "incident"
    assert payload["source_quality"] == "tier_3"
    assert payload["harm_severity"] == pytest.approx(0.8)
