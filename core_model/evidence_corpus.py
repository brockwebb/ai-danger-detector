from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Iterable
import random

from .evidence_schema import EvidenceUnit
from .source_registry import SourceRegistry, SourceStatus


EXCLUDED_SOURCE_STATUSES = {
    SourceStatus.QUARANTINED,
    SourceStatus.DEPRECATED,
    SourceStatus.REMOVED,
}


class EvidenceCorpus:
    def __init__(self, source_registry: SourceRegistry) -> None:
        self.source_registry = source_registry
        self._evidence: dict[str, EvidenceUnit] = {}

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._evidence))

    @property
    def evidence(self) -> tuple[EvidenceUnit, ...]:
        return tuple(self._evidence[evidence_id] for evidence_id in self.evidence_ids)

    def add(self, unit: EvidenceUnit) -> EvidenceUnit:
        if unit.evidence_id in self._evidence:
            raise ValueError(f"evidence_id already exists: {unit.evidence_id}")
        self.source_registry.get_source(unit.source_id)
        self._evidence[unit.evidence_id] = unit
        return unit

    def get(self, evidence_id: str) -> EvidenceUnit:
        try:
            return self._evidence[evidence_id]
        except KeyError as exc:
            raise KeyError(f"unknown evidence_id: {evidence_id}") from exc

    def calibration_evidence(
        self, *, include_experimental: bool = False
    ) -> tuple[EvidenceUnit, ...]:
        eligible: list[EvidenceUnit] = []
        for unit in self.evidence:
            source = self.source_registry.get_source(unit.source_id)
            if source.status in EXCLUDED_SOURCE_STATUSES:
                continue
            if source.status is SourceStatus.EXPERIMENTAL and not include_experimental:
                continue
            if not unit.is_calibration_eligible:
                continue
            eligible.append(unit)
        return tuple(eligible)

    def feature_rows(
        self, evidence: Iterable[EvidenceUnit] | None = None
    ) -> list[dict]:
        units = tuple(evidence) if evidence is not None else self.evidence
        rows: list[dict] = []
        for unit in units:
            source = self.source_registry.get_source(unit.source_id)
            row = unit.to_feature_row()
            row["source_status"] = source.status.value
            row["source_type"] = source.source_type.value
            row["registry_quality_tier"] = source.quality_tier.value
            rows.append(row)
        return rows

    def coverage_summary(
        self, evidence: Iterable[EvidenceUnit] | None = None
    ) -> dict[str, dict[str, int]]:
        units = tuple(evidence) if evidence is not None else self.evidence
        by_domain = Counter(unit.domain for unit in units)
        by_task_type = Counter(unit.task_type for unit in units)
        by_source_id = Counter(unit.source_id for unit in units)
        by_evidence_type = Counter(unit.evidence_type.value for unit in units)
        by_quality_tier = Counter(unit.source_quality.value for unit in units)
        by_source_status = Counter(
            self.source_registry.get_source(unit.source_id).status.value
            for unit in units
        )
        return {
            "by_domain": dict(sorted(by_domain.items())),
            "by_task_type": dict(sorted(by_task_type.items())),
            "by_source_id": dict(sorted(by_source_id.items())),
            "by_evidence_type": dict(sorted(by_evidence_type.items())),
            "by_quality_tier": dict(sorted(by_quality_tier.items())),
            "by_source_status": dict(sorted(by_source_status.items())),
        }
