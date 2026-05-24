import pytest

from core_model.evidence_schema import EvidenceQualityTier
from core_model.source_registry import (
    SourceRegistry,
    SourceStatus,
    SourceType,
)


def test_source_registry_adds_experimental_source():
    registry = SourceRegistry()

    source = registry.add_source(
        source_id="src-001",
        source_name="Public incident review set",
        source_type=SourceType.INCIDENT_REPOSITORY,
        owner_or_publisher="ADD test",
        license_or_access="public",
        update_cadence="periodic",
        coverage=("health", "law"),
        known_biases=("severity bias",),
        quality_tier=EvidenceQualityTier.TIER_3,
    )

    assert source.status is SourceStatus.EXPERIMENTAL
    assert registry.get_source("src-001") == source
    assert registry.audit_events[-1].action == "add_source"


def test_source_registry_prevents_duplicate_source_ids():
    registry = SourceRegistry()
    registry.add_source(
        source_id="src-001",
        source_name="A",
        source_type=SourceType.CASE_SET,
        owner_or_publisher="ADD",
        license_or_access="private",
        update_cadence="one-time",
        coverage=("software",),
        known_biases=(),
        quality_tier=EvidenceQualityTier.TIER_2,
    )

    with pytest.raises(ValueError, match="already exists"):
        registry.add_source(
            source_id="src-001",
            source_name="B",
            source_type=SourceType.CASE_SET,
            owner_or_publisher="ADD",
            license_or_access="private",
            update_cadence="one-time",
            coverage=("software",),
            known_biases=(),
            quality_tier=EvidenceQualityTier.TIER_2,
        )


def test_source_registry_requires_reason_for_non_active_status_change():
    registry = SourceRegistry()
    registry.add_source(
        source_id="src-001",
        source_name="A",
        source_type=SourceType.CASE_SET,
        owner_or_publisher="ADD",
        license_or_access="private",
        update_cadence="one-time",
        coverage=("software",),
        known_biases=(),
        quality_tier=EvidenceQualityTier.TIER_2,
    )

    with pytest.raises(ValueError, match="reason"):
        registry.update_status("src-001", SourceStatus.QUARANTINED)


def test_source_registry_tracks_status_audit_trail():
    registry = SourceRegistry()
    registry.add_source(
        source_id="src-001",
        source_name="A",
        source_type=SourceType.CASE_SET,
        owner_or_publisher="ADD",
        license_or_access="private",
        update_cadence="one-time",
        coverage=("software",),
        known_biases=(),
        quality_tier=EvidenceQualityTier.TIER_2,
    )

    registry.update_status(
        "src-001", SourceStatus.ACTIVE, reason="passed admission review"
    )
    registry.update_status(
        "src-001", SourceStatus.QUARANTINED, reason="degraded holdout calibration"
    )

    source = registry.get_source("src-001")
    assert source.status is SourceStatus.QUARANTINED
    assert source.removal_reason == "degraded holdout calibration"
    assert [event.action for event in registry.audit_events] == [
        "add_source",
        "update_status",
        "update_status",
    ]
