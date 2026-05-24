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
