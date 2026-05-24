# ADD Evidence Data I/O Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standard-library data I/O layer that loads source registry JSON and evidence JSONL into the existing ADD evidence corpus.

**Architecture:** Create `core_model/evidence_io.py` as a focused adapter between local files and the existing `SourceRegistry`, `EvidenceUnit`, and `EvidenceCorpus` objects. The loader converts string enum values, wraps malformed data in `EvidenceLoadError` with path and line context, and leaves modeling/calibration out of scope.

**Tech Stack:** Python 3.12-3.14, standard-library `json` and `pathlib`, pytest.

---

## File Structure

- Create `core_model/evidence_io.py`: source/evidence file loading, enum coercion, structured load errors, corpus assembly.
- Modify `core_model/__init__.py`: export `EvidenceLoadError`, `load_source_registry`, `load_evidence_units`, and `load_corpus`.
- Create `tests/test_evidence_io.py`: behavior tests for valid loading and failure context.
- Modify `README.md`: include `evidence_io.py` and `test_evidence_io.py` in project structure.
- Modify `docs/evidence-data-architecture.md`: document local JSON/JSONL loader as the first file-backed data path.

---

### Task 1: Source Registry JSON Loader

**Files:**
- Create: `tests/test_evidence_io.py`
- Create: `core_model/evidence_io.py`

- [ ] **Step 1: Write failing tests for source registry loading and malformed source JSON**

Create `tests/test_evidence_io.py` with:

```python
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
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_evidence_io.py -v
```

Expected: FAIL because `core_model.evidence_io` does not exist.

- [ ] **Step 3: Implement minimal source registry loading**

Create `core_model/evidence_io.py` with:

```python
from __future__ import annotations

import json
from enum import Enum
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Iterable

from .evidence_corpus import EvidenceCorpus
from .evidence_schema import (
    EvidenceQualityTier,
    EvidenceType,
    EvidenceUnit,
    OutcomeLabel,
    OversightLabel,
    UserExpertise,
)
from .source_registry import SourceRegistry, SourceStatus, SourceType


class EvidenceLoadError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        path: str | Path,
        line_number: int | None = None,
    ) -> None:
        self.message = message
        self.path = Path(path)
        self.line_number = line_number
        super().__init__(str(self))

    def __str__(self) -> str:
        location = str(self.path)
        if self.line_number is not None:
            location = f"{location}: line {self.line_number}"
        return f"{location}: {self.message}"


def _load_json_array(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvidenceLoadError(str(exc), path=path) from exc
    except JSONDecodeError as exc:
        raise EvidenceLoadError(exc.msg, path=path, line_number=exc.lineno) from exc

    if not isinstance(loaded, list):
        raise EvidenceLoadError("expected a JSON array", path=path)
    for index, entry in enumerate(loaded, start=1):
        if not isinstance(entry, dict):
            raise EvidenceLoadError(
                f"expected source entry {index} to be an object",
                path=path,
            )
    return loaded


def _coerce_enum(
    enum_type: type[Enum],
    value: Any,
    field_name: str,
    *,
    path: str | Path,
    line_number: int | None = None,
) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except ValueError as exc:
        raise EvidenceLoadError(
            f"invalid {field_name}: {value!r}",
            path=path,
            line_number=line_number,
        ) from exc


def _add_source_entry(
    registry: SourceRegistry,
    entry: dict[str, Any],
    *,
    path: str | Path,
) -> None:
    try:
        source = registry.add_source(
            source_id=entry["source_id"],
            source_name=entry["source_name"],
            source_type=_coerce_enum(
                SourceType, entry["source_type"], "source_type", path=path
            ),
            owner_or_publisher=entry["owner_or_publisher"],
            license_or_access=entry["license_or_access"],
            update_cadence=entry["update_cadence"],
            coverage=entry["coverage"],
            known_biases=entry["known_biases"],
            quality_tier=_coerce_enum(
                EvidenceQualityTier,
                entry["quality_tier"],
                "quality_tier",
                path=path,
            ),
        )
    except KeyError as exc:
        raise EvidenceLoadError(f"missing source field: {exc.args[0]}", path=path) from exc
    except ValueError as exc:
        raise EvidenceLoadError(str(exc), path=path) from exc

    if "status" in entry:
        status = _coerce_enum(SourceStatus, entry["status"], "status", path=path)
        if status is not source.status or entry.get("status_reason"):
            try:
                registry.update_status(
                    source.source_id,
                    status,
                    reason=entry.get("status_reason"),
                )
            except ValueError as exc:
                raise EvidenceLoadError(str(exc), path=path) from exc


def load_source_registry(path: str | Path) -> SourceRegistry:
    registry = SourceRegistry()
    for entry in _load_json_array(path):
        _add_source_entry(registry, entry, path=path)
    return registry
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_evidence_io.py -v
```

Expected: both source loader tests pass.

- [ ] **Step 5: Commit source loader**

```bash
git add core_model/evidence_io.py tests/test_evidence_io.py
git commit -m "feat: add source registry data loader"
```

---

### Task 2: Evidence JSONL and Corpus Loader

**Files:**
- Modify: `tests/test_evidence_io.py`
- Modify: `core_model/evidence_io.py`
- Modify: `core_model/__init__.py`

- [ ] **Step 1: Add failing tests for evidence loading, corpus assembly, error context, and snapshots**

Append to `tests/test_evidence_io.py`:

```python
from core_model.evidence_io import load_corpus, load_evidence_units


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
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_evidence_io.py -v
```

Expected: FAIL because `load_evidence_units` and `load_corpus` are not implemented.

- [ ] **Step 3: Implement evidence and corpus loading**

Append these functions to `core_model/evidence_io.py`:

```python
def _evidence_unit_from_dict(
    entry: dict[str, Any],
    *,
    path: str | Path,
    line_number: int,
) -> EvidenceUnit:
    try:
        return EvidenceUnit(
            evidence_id=entry["evidence_id"],
            source_id=entry["source_id"],
            evidence_type=_coerce_enum(
                EvidenceType,
                entry["evidence_type"],
                "evidence_type",
                path=path,
                line_number=line_number,
            ),
            collection_date=entry["collection_date"],
            event_date=entry.get("event_date"),
            domain=entry["domain"],
            task_type=entry["task_type"],
            model_family=entry["model_family"],
            model_version=entry["model_version"],
            user_expertise=_coerce_enum(
                UserExpertise,
                entry["user_expertise"],
                "user_expertise",
                path=path,
                line_number=line_number,
            ),
            governance_context=entry["governance_context"],
            outcome_label=_coerce_enum(
                OutcomeLabel,
                entry["outcome_label"],
                "outcome_label",
                path=path,
                line_number=line_number,
            ),
            oversight_label=_coerce_enum(
                OversightLabel,
                entry["oversight_label"],
                "oversight_label",
                path=path,
                line_number=line_number,
            ),
            harm_severity=entry["harm_severity"],
            detectability=entry["detectability"],
            reversibility=entry["reversibility"],
            verification_burden=entry["verification_burden"],
            workflow_path=entry.get("workflow_path", ()),
            confidence=entry.get("confidence", 0.5),
            source_quality=_coerce_enum(
                EvidenceQualityTier,
                entry.get("source_quality", "tier_3"),
                "source_quality",
                path=path,
                line_number=line_number,
            ),
            bias_notes=entry.get("bias_notes", ()),
            relevance_limits=entry.get("relevance_limits", ()),
            optional_fields=entry.get("optional_fields", {}),
        )
    except KeyError as exc:
        raise EvidenceLoadError(
            f"missing evidence field: {exc.args[0]}",
            path=path,
            line_number=line_number,
        ) from exc
    except ValueError as exc:
        if isinstance(exc, EvidenceLoadError):
            raise
        raise EvidenceLoadError(str(exc), path=path, line_number=line_number) from exc


def _iter_evidence_units(path: str | Path) -> Iterable[tuple[int, EvidenceUnit]]:
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvidenceLoadError(str(exc), path=path) from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            loaded = json.loads(line)
        except JSONDecodeError as exc:
            raise EvidenceLoadError(
                exc.msg,
                path=path,
                line_number=line_number,
            ) from exc
        if not isinstance(loaded, dict):
            raise EvidenceLoadError(
                "expected evidence line to be an object",
                path=path,
                line_number=line_number,
            )
        yield line_number, _evidence_unit_from_dict(
            loaded,
            path=path,
            line_number=line_number,
        )


def load_evidence_units(path: str | Path) -> tuple[EvidenceUnit, ...]:
    return tuple(unit for _, unit in _iter_evidence_units(path))


def load_corpus(source_path: str | Path, evidence_path: str | Path) -> EvidenceCorpus:
    registry = load_source_registry(source_path)
    corpus = EvidenceCorpus(registry)
    for line_number, unit in _iter_evidence_units(evidence_path):
        try:
            corpus.add(unit)
        except (KeyError, ValueError) as exc:
            raise EvidenceLoadError(
                str(exc),
                path=evidence_path,
                line_number=line_number,
            ) from exc
    return corpus
```

Remove unused imports from `core_model/evidence_io.py` after the tests pass.

Update `core_model/__init__.py`:

```python
from .evidence_io import (
    EvidenceLoadError,
    load_corpus,
    load_evidence_units,
    load_source_registry,
)
```

Add these entries to `__all__`:

```python
"EvidenceLoadError",
"load_corpus",
"load_evidence_units",
"load_source_registry",
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_evidence_io.py -v
```

Expected: all evidence I/O tests pass.

- [ ] **Step 5: Run related backend tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_evidence_io.py tests/test_evidence_corpus.py tests/test_source_registry.py tests/test_evidence_schema.py -v
```

Expected: all selected backend tests pass.

- [ ] **Step 6: Commit evidence and corpus loader**

```bash
git add core_model tests
git commit -m "feat: add evidence corpus data loader"
```

---

### Task 3: Documentation and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/evidence-data-architecture.md`

- [ ] **Step 1: Update README project structure**

In `README.md`, add:

```text
|   |-- evidence_io.py              # Local JSON/JSONL evidence loading
```

Under tests, add:

```text
|   |-- test_evidence_io.py         # Evidence data loading tests
```

- [ ] **Step 2: Update evidence architecture reference implementation**

In `docs/evidence-data-architecture.md`, update the implementation path and reference implementation sections to include `core_model/evidence_io.py` and explain that it is the first local file-backed path into the evidence corpus.

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
rg -n "\b(t[i]ck|d[e]er|a[c]orn|f[o]rest)\b" README.md docs core_model tests || true
git status --short --branch
```

Expected:

- pytest passes,
- no overclaim terms are reported,
- no wrong-domain terms are reported,
- status shows only intended docs changes before commit.

- [ ] **Step 4: Commit docs**

```bash
git add README.md docs/evidence-data-architecture.md
git commit -m "docs: document evidence data io"
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

Expected: exports include `EvidenceLoadError`, `load_source_registry`, `load_evidence_units`, and `load_corpus`.

- [ ] **Step 3: Check branch status**

```bash
git status --short --branch
```

Expected: working tree is clean on the feature branch before merge.
