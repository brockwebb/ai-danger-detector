import json

import pytest

from core_model.evidence_io import (
    EvidenceLoadError,
    load_corpus,
    load_evidence_units,
    load_source_registry,
)
from core_model.evidence_schema import EvidenceQualityTier
from core_model.source_registry import SourceStatus, SourceType


def _source_record(**overrides):
    values = {
        "source_id": "src-active",
        "source_name": "Active adjudicated set",
        "source_type": "case_set",
        "owner_or_publisher": "ADD",
        "license_or_access": "private",
        "update_cadence": "one-time",
        "coverage": ["health"],
        "known_biases": [],
        "quality_tier": "tier_1",
        "status": "active",
        "status_reason": "approved",
    }
    values.update(overrides)
    return values


def test_load_source_registry_loads_statused_sources(tmp_path):
    source_path = tmp_path / "sources.json"
    source_path.write_text(
        json.dumps(
            [
                _source_record(),
                _source_record(
                    source_id="src-experimental",
                    source_name="Experimental incident feed",
                    source_type="incident_repository",
                    coverage=["software"],
                    known_biases=["severity bias"],
                    quality_tier="tier_3",
                    status="experimental",
                ),
            ]
        ),
        encoding="utf-8",
    )

    registry = load_source_registry(source_path)

    active = registry.get_source("src-active")
    experimental = registry.get_source("src-experimental")
    assert active.status is SourceStatus.ACTIVE
    assert active.quality_tier is EvidenceQualityTier.TIER_1
    assert active.source_type is SourceType.CASE_SET
    assert experimental.status is SourceStatus.EXPERIMENTAL
    assert experimental.quality_tier is EvidenceQualityTier.TIER_3
    assert experimental.source_type is SourceType.INCIDENT_REPOSITORY


def test_load_source_registry_wraps_malformed_json(tmp_path):
    source_path = tmp_path / "sources.json"
    source_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(EvidenceLoadError, match="sources.json"):
        load_source_registry(source_path)


def _evidence_record(**overrides):
    values = {
        "evidence_id": "case-001",
        "source_id": "src-active",
        "evidence_type": "case_review",
        "collection_date": "2026-05-24",
        "event_date": "2026-05-20",
        "domain": "health",
        "task_type": "medical symptom advice",
        "model_family": "unknown",
        "model_version": "unknown",
        "user_expertise": "non_expert",
        "governance_context": "informal",
        "outcome_label": "harm",
        "oversight_label": "expert_review_required",
        "harm_severity": 0.8,
        "detectability": 0.2,
        "reversibility": 0.3,
        "verification_burden": 0.9,
        "workflow_path": ["S0", "S1", "S2", "S6", "S8"],
        "confidence": 0.7,
        "source_quality": "tier_1",
        "bias_notes": [],
        "relevance_limits": [],
    }
    values.update(overrides)
    return values


def _write_sources(tmp_path, records=None):
    source_path = tmp_path / "sources.json"
    source_path.write_text(
        json.dumps(records or [_source_record()]),
        encoding="utf-8",
    )
    return source_path


def _write_evidence(tmp_path, records):
    evidence_path = tmp_path / "evidence.jsonl"
    evidence_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return evidence_path


def test_load_evidence_units_loads_jsonl_records(tmp_path):
    evidence_path = _write_evidence(tmp_path, [_evidence_record()])

    units = load_evidence_units(evidence_path)

    assert len(units) == 1
    assert units[0].evidence_id == "case-001"
    assert units[0].workflow_path == ("S0", "S1", "S2", "S6", "S8")


def test_load_evidence_units_wraps_malformed_jsonl_with_line_context(tmp_path):
    evidence_path = tmp_path / "evidence.jsonl"
    evidence_path.write_text("\n{not json\n", encoding="utf-8")

    with pytest.raises(EvidenceLoadError, match="line 2"):
        load_evidence_units(evidence_path)


def test_load_corpus_links_evidence_to_registered_sources(tmp_path):
    source_path = _write_sources(tmp_path)
    evidence_path = _write_evidence(tmp_path, [_evidence_record()])

    corpus = load_corpus(source_path, evidence_path)

    assert corpus.evidence_ids == ("case-001",)
    assert corpus.feature_rows()[0]["source_status"] == "active"


def test_load_corpus_rejects_evidence_with_missing_source(tmp_path):
    source_path = _write_sources(tmp_path)
    evidence_path = _write_evidence(
        tmp_path,
        [_evidence_record(source_id="missing-source")],
    )

    with pytest.raises(EvidenceLoadError, match="missing-source"):
        load_corpus(source_path, evidence_path)


def test_loaded_corpus_can_create_snapshot(tmp_path):
    source_path = _write_sources(tmp_path)
    evidence_path = _write_evidence(tmp_path, [_evidence_record()])

    corpus = load_corpus(source_path, evidence_path)
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
