from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from .evidence_schema import EvidenceQualityTier


class SourceType(str, Enum):
    INCIDENT_REPOSITORY = "incident_repository"
    BENCHMARK_SUITE = "benchmark_suite"
    EXPERT_PANEL = "expert_panel"
    DEPLOYMENT_LOG = "deployment_log"
    USER_STUDY = "user_study"
    AUDIT = "audit"
    LITERATURE_REVIEW = "literature_review"
    CASE_SET = "case_set"
    SYNTHETIC = "synthetic"


class SourceStatus(str, Enum):
    ACTIVE = "active"
    EXPERIMENTAL = "experimental"
    QUARANTINED = "quarantined"
    DEPRECATED = "deprecated"
    REMOVED = "removed"


@dataclass(frozen=True)
class EvidenceSource:
    source_id: str
    source_name: str
    source_type: SourceType
    owner_or_publisher: str
    license_or_access: str
    update_cadence: str
    coverage: tuple[str, ...]
    known_biases: tuple[str, ...]
    quality_tier: EvidenceQualityTier
    status: SourceStatus = SourceStatus.EXPERIMENTAL
    removal_reason: str | None = None


@dataclass(frozen=True)
class SourceAuditEvent:
    source_id: str
    action: str
    reason: str | None
    status: SourceStatus


def _text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _tuple(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(str(value).strip() for value in values if str(value).strip())


class SourceRegistry:
    def __init__(self) -> None:
        self._sources: dict[str, EvidenceSource] = {}
        self._audit_events: list[SourceAuditEvent] = []

    @property
    def audit_events(self) -> tuple[SourceAuditEvent, ...]:
        return tuple(self._audit_events)

    def add_source(
        self,
        *,
        source_id: str,
        source_name: str,
        source_type: SourceType,
        owner_or_publisher: str,
        license_or_access: str,
        update_cadence: str,
        coverage: tuple[str, ...] | list[str],
        known_biases: tuple[str, ...] | list[str],
        quality_tier: EvidenceQualityTier,
    ) -> EvidenceSource:
        source_id = _text(source_id, "source_id")
        if source_id in self._sources:
            raise ValueError(f"source_id already exists: {source_id}")

        source = EvidenceSource(
            source_id=source_id,
            source_name=_text(source_name, "source_name"),
            source_type=source_type,
            owner_or_publisher=_text(owner_or_publisher, "owner_or_publisher"),
            license_or_access=_text(license_or_access, "license_or_access"),
            update_cadence=_text(update_cadence, "update_cadence"),
            coverage=_tuple(coverage),
            known_biases=_tuple(known_biases),
            quality_tier=quality_tier,
        )
        self._sources[source_id] = source
        self._audit_events.append(
            SourceAuditEvent(source_id, "add_source", None, source.status)
        )
        return source

    def get_source(self, source_id: str) -> EvidenceSource:
        try:
            return self._sources[source_id]
        except KeyError as exc:
            raise KeyError(f"unknown source_id: {source_id}") from exc

    def update_status(
        self,
        source_id: str,
        status: SourceStatus,
        *,
        reason: str | None = None,
    ) -> EvidenceSource:
        if (
            status
            in {
                SourceStatus.QUARANTINED,
                SourceStatus.DEPRECATED,
                SourceStatus.REMOVED,
            }
            and not reason
        ):
            raise ValueError(
                "reason is required when quarantining, deprecating, or removing a source"
            )

        source = self.get_source(source_id)
        updated = replace(
            source,
            status=status,
            removal_reason=reason if status is not SourceStatus.ACTIVE else None,
        )
        self._sources[source_id] = updated
        self._audit_events.append(
            SourceAuditEvent(source_id, "update_status", reason, status)
        )
        return updated
