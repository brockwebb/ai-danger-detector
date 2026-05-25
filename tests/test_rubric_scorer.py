from core_model.evidence_schema import (
    EvidenceQualityTier,
    EvidenceType,
    EvidenceUnit,
    OutcomeLabel,
    OversightLabel,
    UserExpertise,
)
from core_model.rubric_scorer import score_evidence_unit


def _evidence(**overrides):
    values = {
        "evidence_id": "case-001",
        "source_id": "src-active",
        "evidence_type": EvidenceType.CASE_REVIEW,
        "collection_date": "2026-05-24",
        "event_date": "2026-05-20",
        "domain": "creative",
        "task_type": "draft low-stakes text",
        "model_family": "unknown",
        "model_version": "unknown",
        "user_expertise": UserExpertise.TRAINED,
        "governance_context": "ordinary user review",
        "outcome_label": OutcomeLabel.BENIGN_USE,
        "oversight_label": OversightLabel.CASUAL_EXPLORATORY,
        "harm_severity": 0.05,
        "detectability": 0.9,
        "reversibility": 0.95,
        "verification_burden": 0.1,
        "workflow_path": ("S0", "S1", "S3", "S7"),
        "confidence": 0.7,
        "source_quality": EvidenceQualityTier.TIER_1,
        "bias_notes": (),
        "relevance_limits": (),
    }
    values.update(overrides)
    return EvidenceUnit(**values)


def test_low_consequence_case_scores_to_casual_exploratory():
    result = score_evidence_unit(_evidence())

    assert 0.0 <= result.score < 0.2
    assert result.band is OversightLabel.CASUAL_EXPLORATORY
    assert "provisional" in " ".join(result.assumptions)


def test_increasing_harm_increases_score():
    low = score_evidence_unit(_evidence(harm_severity=0.1))
    high = score_evidence_unit(_evidence(harm_severity=0.9))

    assert high.score > low.score


def test_lower_detectability_increases_score():
    easy = score_evidence_unit(_evidence(detectability=0.9))
    hard = score_evidence_unit(_evidence(detectability=0.1))

    assert hard.score > easy.score


def test_lower_source_quality_adds_uncertainty_driver():
    strong = score_evidence_unit(_evidence(source_quality=EvidenceQualityTier.TIER_1))
    weak = score_evidence_unit(_evidence(source_quality=EvidenceQualityTier.TIER_4))

    assert weak.score > strong.score
    assert "source uncertainty" in weak.drivers
