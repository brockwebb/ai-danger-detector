# ADD Evidence Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement an evidence corpus layer that validates evidence against known sources, filters calibration-ready records, summarizes coverage, creates deterministic data splits, and records reproducible data snapshots.

**Architecture:** Add `core_model/evidence_corpus.py` with `EvidenceCorpus`, `EvidenceSplit`, and `DataSnapshot`. The corpus depends on the existing evidence schema and source registry modules; it does not ingest external data or train models. Tests define corpus behavior before implementation.

**Tech Stack:** Python 3.12-3.14, dataclasses, standard-library random/date, pytest.

---

## File Structure

- Create `core_model/evidence_corpus.py`: corpus storage, validation, filtering, feature rows, coverage summaries, splits, and snapshots.
- Modify `core_model/__init__.py`: export corpus API.
- Create `tests/test_evidence_corpus.py`: behavior tests for corpus mechanics.
- Modify `README.md`: include `evidence_corpus.py` in project structure.
- Modify `docs/evidence-data-architecture.md`: note the corpus reference implementation.

---

### Task 1: Corpus Validation and Storage

**Files:**
- Create: `tests/test_evidence_corpus.py`
- Create: `core_model/evidence_corpus.py`
- Modify: `core_model/__init__.py`

- [ ] **Step 1: Write failing tests for corpus creation, source validation, and duplicates**

Create `tests/test_evidence_corpus.py` with:

```python
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
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_evidence_corpus.py -v
```

Expected: FAIL because `core_model.evidence_corpus` does not exist.

- [ ] **Step 3: Implement minimal corpus storage**

Create `core_model/evidence_corpus.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import random
from collections import Counter
from typing import Iterable

from .evidence_schema import EvidenceQualityTier, EvidenceUnit
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
```

Update `core_model/__init__.py` to export `EvidenceCorpus`.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_evidence_corpus.py -v
```

Expected: all current corpus tests pass.

- [ ] **Step 5: Commit corpus storage**

```bash
git add core_model tests
git commit -m "feat: add evidence corpus storage"
```

---

### Task 2: Calibration Filtering, Feature Rows, and Coverage Summaries

**Files:**
- Modify: `tests/test_evidence_corpus.py`
- Modify: `core_model/evidence_corpus.py`

- [ ] **Step 1: Add failing tests for filtering, features, and summaries**

Append to `tests/test_evidence_corpus.py`:

```python
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
    assert [unit.evidence_id for unit in corpus.calibration_evidence(include_experimental=True)] == [
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
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_evidence_corpus.py -v
```

Expected: FAIL because filtering, feature rows, or coverage summaries are not implemented.

- [ ] **Step 3: Implement filtering, feature rows, and coverage summaries**

Add to `EvidenceCorpus`:

```python
    def calibration_evidence(self, *, include_experimental: bool = False) -> tuple[EvidenceUnit, ...]:
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

    def feature_rows(self, evidence: Iterable[EvidenceUnit] | None = None) -> list[dict]:
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

    def coverage_summary(self, evidence: Iterable[EvidenceUnit] | None = None) -> dict[str, dict[str, int]]:
        units = tuple(evidence) if evidence is not None else self.evidence
        by_domain = Counter(unit.domain for unit in units)
        by_task_type = Counter(unit.task_type for unit in units)
        by_source_id = Counter(unit.source_id for unit in units)
        by_evidence_type = Counter(unit.evidence_type.value for unit in units)
        by_quality_tier = Counter(unit.source_quality.value for unit in units)
        by_source_status = Counter(
            self.source_registry.get_source(unit.source_id).status.value for unit in units
        )
        return {
            "by_domain": dict(sorted(by_domain.items())),
            "by_task_type": dict(sorted(by_task_type.items())),
            "by_source_id": dict(sorted(by_source_id.items())),
            "by_evidence_type": dict(sorted(by_evidence_type.items())),
            "by_quality_tier": dict(sorted(by_quality_tier.items())),
            "by_source_status": dict(sorted(by_source_status.items())),
        }
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_evidence_corpus.py -v
```

Expected: all current corpus tests pass.

- [ ] **Step 5: Commit filtering and summaries**

```bash
git add core_model tests
git commit -m "feat: add evidence corpus filtering and summaries"
```

---

### Task 3: Deterministic Splits and Data Snapshots

**Files:**
- Modify: `tests/test_evidence_corpus.py`
- Modify: `core_model/evidence_corpus.py`
- Modify: `core_model/__init__.py`

- [ ] **Step 1: Add failing tests for splits and snapshots**

Append to `tests/test_evidence_corpus.py`:

```python
def test_create_split_is_deterministic_for_same_seed():
    corpus = EvidenceCorpus(_registry())
    for index in range(10):
        corpus.add(_evidence(f"case-{index:03d}", "src-active"))

    first = corpus.create_split(seed=42, train_ratio=0.6, calibration_ratio=0.2, holdout_ratio=0.2)
    second = corpus.create_split(seed=42, train_ratio=0.6, calibration_ratio=0.2, holdout_ratio=0.2)

    assert first == second
    assert len(first.train_ids) == 6
    assert len(first.calibration_ids) == 2
    assert len(first.holdout_ids) == 2


def test_create_split_rejects_invalid_ratios():
    corpus = EvidenceCorpus(_registry())
    corpus.add(_evidence())

    with pytest.raises(ValueError, match="sum to 1"):
        corpus.create_split(seed=42, train_ratio=0.5, calibration_ratio=0.2, holdout_ratio=0.2)


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
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_evidence_corpus.py -v
```

Expected: FAIL because split and snapshot APIs are not implemented.

- [ ] **Step 3: Implement `EvidenceSplit`, `DataSnapshot`, split creation, and snapshots**

Add above `EvidenceCorpus`:

```python
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
```

Add to `EvidenceCorpus`:

```python
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
        calibration_ids = tuple(sorted(evidence_ids[train_count : train_count + calibration_count]))
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
```

Update `core_model/__init__.py` to export `DataSnapshot` and `EvidenceSplit`.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_evidence_corpus.py -v
```

Expected: all corpus tests pass.

- [ ] **Step 5: Commit splits and snapshots**

```bash
git add core_model tests
git commit -m "feat: add evidence corpus splits and snapshots"
```

---

### Task 4: Documentation and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/evidence-data-architecture.md`

- [ ] **Step 1: Update README project structure**

Add `evidence_corpus.py` under `core_model/` in `README.md`:

```text
|   |-- evidence_corpus.py          # Evidence corpus, summaries, splits, snapshots
```

- [ ] **Step 2: Update evidence architecture reference implementation**

In `docs/evidence-data-architecture.md`, update the reference implementation paragraph to include `core_model/evidence_corpus.py` and describe it as the corpus bridge between evidence records and future calibration/model runs.

- [ ] **Step 3: Run full verification**

Run:

```bash
.venv/bin/python -m pytest -v
.venv/bin/python - <<'PY'
from pathlib import Path

terms = [
    "scientifically " + "validated",
    "objective " + "detector",
    "guarantees " + "safety",
    "safe " + "to " + "use",
]
roots = [Path("README.md"), Path("docs"), Path("core_model"), Path("tests")]
for root in roots:
    paths = [root] if root.is_file() else root.rglob("*")
    for path in paths:
        if path.is_file() and path.suffix in {".md", ".py"}:
            text = path.read_text(encoding="utf-8")
            for term in terms:
                if term in text:
                    print(f"{path}: contains {term!r}")
PY
git status --short --branch
```

Expected:

- all tests pass,
- no overclaims appear,
- status shows only intended docs changes before commit.

- [ ] **Step 4: Commit docs**

```bash
git add README.md docs/evidence-data-architecture.md
git commit -m "docs: document evidence corpus implementation"
```

---

## Final Verification

- [ ] **Step 1: Run full tests**

```bash
.venv/bin/python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Inspect exports**

```bash
.venv/bin/python - <<'PY'
import core_model
print(core_model.__all__)
PY
```

Expected: exports include `EvidenceCorpus`, `EvidenceSplit`, and `DataSnapshot`.

- [ ] **Step 3: Check status**

```bash
git status --short --branch
```

Expected: working tree is clean on the feature branch before merge.
