import json

import pytest

from core_model.evidence_io import EvidenceLoadError, load_source_registry
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
