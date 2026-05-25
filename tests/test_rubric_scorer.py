from core_model.evidence_schema import (
    EvidenceQualityTier,
    EvidenceType,
    EvidenceUnit,
    OutcomeLabel,
    OversightLabel,
    UserExpertise,
)
from core_model.evidence_corpus import EvidenceCorpus
from core_model.evidence_io import load_corpus
from core_model.source_registry import SourceRegistry, SourceStatus, SourceType
from core_model.rubric_scorer import score_evidence_unit, score_feature_row


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


def test_high_harm_low_detectability_escalates_to_expert_review():
    result = score_evidence_unit(
        _evidence(
            domain="health",
            harm_severity=0.85,
            detectability=0.2,
            reversibility=0.5,
            verification_burden=0.8,
            user_expertise=UserExpertise.NON_EXPERT,
            governance_context="no expert review",
        )
    )

    assert result.band in {
        OversightLabel.EXPERT_REVIEW_REQUIRED,
        OversightLabel.EXPERT_LED_OR_NO_AUTONOMOUS_USE,
    }
    assert any("high harm" in driver for driver in result.drivers)


def test_extreme_harm_low_detection_and_low_reversibility_escalates_to_expert_led():
    result = score_evidence_unit(
        _evidence(
            harm_severity=0.95,
            detectability=0.2,
            reversibility=0.2,
            verification_burden=0.95,
            user_expertise=UserExpertise.NON_EXPERT,
            governance_context="no review",
        )
    )

    assert result.band is OversightLabel.EXPERT_LED_OR_NO_AUTONOMOUS_USE


def test_feature_rows_can_be_scored():
    registry = SourceRegistry()
    registry.add_source(
        source_id="src-active",
        source_name="Active cases",
        source_type=SourceType.CASE_SET,
        owner_or_publisher="ADD",
        license_or_access="private",
        update_cadence="one-time",
        coverage=("creative",),
        known_biases=(),
        quality_tier=EvidenceQualityTier.TIER_1,
    )
    registry.update_status("src-active", SourceStatus.ACTIVE, reason="approved")
    corpus = EvidenceCorpus(registry)
    corpus.add(_evidence())

    result = score_feature_row(corpus.feature_rows()[0])

    assert result.band is OversightLabel.CASUAL_EXPLORATORY
    assert result.factor_scores["harm_severity"] == 0.05


def test_example_corpus_scores_high_stakes_above_creative_case():
    corpus = load_corpus("data/examples/sources.json", "data/examples/evidence.jsonl")

    scores = {
        unit.domain: score_evidence_unit(unit)
        for unit in corpus.evidence
    }

    assert scores["health"].score > scores["creative"].score
    assert scores["finance"].score > scores["education"].score
    assert scores["health"].band is OversightLabel.EXPERT_LED_OR_NO_AUTONOMOUS_USE
