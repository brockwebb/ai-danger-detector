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


@dataclass(frozen=True)
class EvidenceSplit:
    train_ids: tuple[str, ...]
    calibration_ids: tuple[str, ...]
    holdout_ids: tuple[str, ...]


@dataclass(frozen=True)
class DataSnapshot:
    snapshot_id: str
    schema_version: str
    source_registry_version: str
    feature_transformation_version: str
    created_date: str
    evidence_count: int
    source_count: int
    included_evidence_ids: tuple[str, ...]
    included_source_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "schema_version": self.schema_version,
            "source_registry_version": self.source_registry_version,
            "feature_transformation_version": self.feature_transformation_version,
            "created_date": self.created_date,
            "evidence_count": self.evidence_count,
            "source_count": self.source_count,
            "included_evidence_ids": self.included_evidence_ids,
            "included_source_ids": self.included_source_ids,
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

    def create_split(
        self,
        *,
        seed: int,
        train_ratio: float,
        calibration_ratio: float,
        holdout_ratio: float,
        calibration_only: bool = False,
    ) -> EvidenceSplit:
        ratios = (train_ratio, calibration_ratio, holdout_ratio)
        if any(ratio <= 0 for ratio in ratios):
            raise ValueError("split ratios must be positive")
        if abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1")

        units = self.calibration_evidence() if calibration_only else self.evidence
        evidence_ids = [unit.evidence_id for unit in units]
        if not evidence_ids:
            raise ValueError("cannot split an empty corpus")

        rng = random.Random(seed)
        rng.shuffle(evidence_ids)

        total = len(evidence_ids)
        train_count = int(total * train_ratio)
        calibration_count = int(total * calibration_ratio)
        holdout_count = total - train_count - calibration_count
        if train_count == 0 or calibration_count == 0 or holdout_count == 0:
            raise ValueError("each split must contain at least one record")

        train_ids = tuple(sorted(evidence_ids[:train_count]))
        calibration_ids = tuple(
            sorted(evidence_ids[train_count : train_count + calibration_count])
        )
        holdout_ids = tuple(sorted(evidence_ids[train_count + calibration_count :]))
        return EvidenceSplit(train_ids, calibration_ids, holdout_ids)

    def create_snapshot(
        self,
        *,
        snapshot_id: str,
        schema_version: str,
        source_registry_version: str,
        feature_transformation_version: str,
        created_date: str | None = None,
        evidence: Iterable[EvidenceUnit] | None = None,
    ) -> DataSnapshot:
        units = tuple(evidence) if evidence is not None else self.evidence
        evidence_ids = tuple(sorted(unit.evidence_id for unit in units))
        source_ids = tuple(sorted({unit.source_id for unit in units}))
        return DataSnapshot(
            snapshot_id=snapshot_id,
            schema_version=schema_version,
            source_registry_version=source_registry_version,
            feature_transformation_version=feature_transformation_version,
            created_date=created_date or date.today().isoformat(),
            evidence_count=len(evidence_ids),
            source_count=len(source_ids),
            included_evidence_ids=evidence_ids,
            included_source_ids=source_ids,
        )
