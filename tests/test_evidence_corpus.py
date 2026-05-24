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


def _registry_with_mixed_sources():
    registry = _registry()
    registry.add_source(
        source_id="src-experimental",
        source_name="Experimental incident feed",
        source_type=SourceType.INCIDENT_REPOSITORY,
        owner_or_publisher="ADD",
        license_or_access="public",
        update_cadence="periodic",
        coverage=("software",),
        known_biases=("severity bias",),
        quality_tier=EvidenceQualityTier.TIER_3,
    )
    registry.add_source(
        source_id="src-quarantined",
        source_name="Quarantined feed",
        source_type=SourceType.INCIDENT_REPOSITORY,
        owner_or_publisher="ADD",
        license_or_access="public",
        update_cadence="periodic",
        coverage=("finance",),
        known_biases=("duplication",),
        quality_tier=EvidenceQualityTier.TIER_4,
    )
    registry.update_status(
        "src-quarantined", SourceStatus.QUARANTINED, reason="unstable labels"
    )
    return registry


def test_calibration_evidence_excludes_experimental_sources_by_default():
    corpus = EvidenceCorpus(_registry_with_mixed_sources())
    corpus.add(_evidence("case-001", "src-active"))
    corpus.add(_evidence("case-002", "src-experimental", domain="software"))

    assert [unit.evidence_id for unit in corpus.calibration_evidence()] == ["case-001"]
    assert [
        unit.evidence_id
        for unit in corpus.calibration_evidence(include_experimental=True)
    ] == [
        "case-001",
        "case-002",
    ]


def test_calibration_evidence_excludes_quarantined_sources_and_records():
    corpus = EvidenceCorpus(_registry_with_mixed_sources())
    corpus.add(_evidence("case-001", "src-active"))
    corpus.add(_evidence("case-002", "src-quarantined", domain="finance"))
    corpus.add(
        _evidence(
            "case-003",
            "src-active",
            source_quality=EvidenceQualityTier.QUARANTINED,
        )
    )

    assert [unit.evidence_id for unit in corpus.calibration_evidence()] == ["case-001"]


def test_feature_rows_include_source_traceability():
    corpus = EvidenceCorpus(_registry())
    corpus.add(_evidence())

    rows = corpus.feature_rows()

    assert rows[0]["evidence_id"] == "case-001"
    assert rows[0]["source_id"] == "src-active"
    assert rows[0]["source_status"] == "active"
    assert rows[0]["source_type"] == "case_set"
    assert rows[0]["registry_quality_tier"] == "tier_1"


def test_coverage_summary_counts_core_dimensions():
    corpus = EvidenceCorpus(_registry_with_mixed_sources())
    corpus.add(_evidence("case-001", "src-active", domain="health", task_type="triage"))
    corpus.add(
        _evidence(
            "case-002",
            "src-experimental",
            domain="software",
            task_type="code review",
            evidence_type=EvidenceType.INCIDENT,
            source_quality=EvidenceQualityTier.TIER_3,
        )
    )

    summary = corpus.coverage_summary()

    assert summary["by_domain"] == {"health": 1, "software": 1}
    assert summary["by_task_type"] == {"code review": 1, "triage": 1}
    assert summary["by_source_id"] == {"src-active": 1, "src-experimental": 1}
    assert summary["by_evidence_type"] == {"case_review": 1, "incident": 1}
    assert summary["by_quality_tier"] == {"tier_1": 1, "tier_3": 1}
    assert summary["by_source_status"] == {"active": 1, "experimental": 1}


def test_create_split_is_deterministic_for_same_seed():
    corpus = EvidenceCorpus(_registry())
    for index in range(10):
        corpus.add(_evidence(f"case-{index:03d}", "src-active"))

    first = corpus.create_split(
        seed=42, train_ratio=0.6, calibration_ratio=0.2, holdout_ratio=0.2
    )
    second = corpus.create_split(
        seed=42, train_ratio=0.6, calibration_ratio=0.2, holdout_ratio=0.2
    )

    assert first == second
    assert len(first.train_ids) == 6
    assert len(first.calibration_ids) == 2
    assert len(first.holdout_ids) == 2


def test_create_split_rejects_invalid_ratios():
    corpus = EvidenceCorpus(_registry())
    corpus.add(_evidence())

    with pytest.raises(ValueError, match="sum to 1"):
        corpus.create_split(
            seed=42, train_ratio=0.5, calibration_ratio=0.2, holdout_ratio=0.2
        )


def test_data_snapshot_records_version_and_inclusion_metadata():
    corpus = EvidenceCorpus(_registry())
    corpus.add(_evidence())

    snapshot = corpus.create_snapshot(
        snapshot_id="snapshot-001",
        schema_version="schema-v1",
        source_registry_version="sources-v1",
        feature_transformation_version="features-v1",
        created_date="2026-05-24",
    )

    assert snapshot.evidence_count == 1
    assert snapshot.source_count == 1
    assert snapshot.included_evidence_ids == ("case-001",)
    assert snapshot.included_source_ids == ("src-active",)
    assert snapshot.to_dict()["snapshot_id"] == "snapshot-001"
