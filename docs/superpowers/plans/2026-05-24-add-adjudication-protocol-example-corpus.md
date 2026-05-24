# ADD Adjudication Protocol and Example Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an adjudication protocol and a tested illustrative example corpus for ADD evidence development.

**Architecture:** Keep the protocol as documentation and the example corpus as local JSON/JSONL data loaded through the existing `core_model.evidence_io` API. Tests validate that the corpus loads, demonstrates label variety, can produce a snapshot, and remains excluded from calibration by default because the source is experimental and illustrative.

**Tech Stack:** Markdown, JSON, JSON Lines, Python 3.12-3.14, pytest, existing ADD evidence loader.

---

## File Structure

- Create `docs/adjudication-protocol.md`: case intake, reviewer roles, label workflow, quality gates, calibration eligibility.
- Create `data/examples/README.md`: example corpus purpose, file format, and limitations.
- Create `data/examples/sources.json`: synthetic source registry entry.
- Create `data/examples/evidence.jsonl`: six fictional example evidence units.
- Create `tests/test_example_corpus.py`: loader and corpus behavior tests.
- Modify `README.md`: add protocol and example data to project structure.
- Modify `docs/evidence-data-architecture.md`: reference protocol and example corpus.
- Modify `docs/validation-agenda.md`: connect retrospective scoring to the adjudication protocol.

---

### Task 1: Adjudication Protocol

**Files:**
- Create: `docs/adjudication-protocol.md`

- [ ] **Step 1: Create adjudication protocol document**

Create `docs/adjudication-protocol.md` with sections for purpose, case intake, reviewer roles, label workflow, scoring anchors, workflow path coding, disagreement handling, quality gates, calibration eligibility, privacy/governance, and reporting.

The document must include these explicit requirements:

```text
The protocol is provisional and does not validate ADD by itself.
Calibration eligibility requires an active source, adequate provenance, reviewed labels, and no quarantine flags.
Synthetic or illustrative cases are excluded from calibration by default.
Reviewer disagreement is preserved as evidence, not erased as noise.
```

- [ ] **Step 2: Scan protocol for prohibited overclaims and placeholders**

Run:

```bash
rg -n "TBD|TODO|FIXME|scientifically[ ]validated|objective[ ]detector|guarantees[ ]safety|safe[ ]to[ ]use|\b(t[i]ck|d[e]er|a[c]orn|f[o]rest)\b" docs/adjudication-protocol.md || true
```

Expected: no matches.

- [ ] **Step 3: Commit protocol**

```bash
git add docs/adjudication-protocol.md
git commit -m "docs: add adjudication protocol"
```

---

### Task 2: Example Corpus Tests

**Files:**
- Create: `tests/test_example_corpus.py`

- [ ] **Step 1: Write failing tests for example corpus files**

Create `tests/test_example_corpus.py` with:

```python
from pathlib import Path

from core_model.evidence_io import load_corpus, load_source_registry
from core_model.evidence_schema import OutcomeLabel, OversightLabel
from core_model.source_registry import SourceStatus, SourceType


EXAMPLE_DIR = Path("data/examples")
SOURCE_PATH = EXAMPLE_DIR / "sources.json"
EVIDENCE_PATH = EXAMPLE_DIR / "evidence.jsonl"


def test_example_corpus_loads_and_has_expected_shape():
    corpus = load_corpus(SOURCE_PATH, EVIDENCE_PATH)

    assert len(corpus.evidence) == 6
    assert corpus.coverage_summary()["by_domain"] == {
        "creative": 1,
        "education": 1,
        "finance": 1,
        "health": 1,
        "law": 1,
        "software": 1,
    }


def test_example_source_is_illustrative_and_not_calibration_ready():
    registry = load_source_registry(SOURCE_PATH)
    source = registry.get_source("src-illustrative-add-cases")
    corpus = load_corpus(SOURCE_PATH, EVIDENCE_PATH)

    assert source.source_type is SourceType.SYNTHETIC
    assert source.status is SourceStatus.EXPERIMENTAL
    assert "fictional examples" in source.known_biases
    assert corpus.calibration_evidence() == ()


def test_example_corpus_contains_outcome_and_oversight_variety():
    corpus = load_corpus(SOURCE_PATH, EVIDENCE_PATH)
    outcomes = {unit.outcome_label for unit in corpus.evidence}
    oversight = {unit.oversight_label for unit in corpus.evidence}

    assert {
        OutcomeLabel.BENIGN_USE,
        OutcomeLabel.CORRECTED_ERROR,
        OutcomeLabel.HARM,
        OutcomeLabel.LOSS,
        OutcomeLabel.NEAR_MISS,
    }.issubset(outcomes)
    assert {
        OversightLabel.CASUAL_EXPLORATORY,
        OversightLabel.ASSISTED_BOUNDED,
        OversightLabel.TRAINED_REVIEW_REQUIRED,
        OversightLabel.EXPERT_REVIEW_REQUIRED,
        OversightLabel.EXPERT_LED_OR_NO_AUTONOMOUS_USE,
    }.issubset(oversight)


def test_example_corpus_can_create_snapshot():
    corpus = load_corpus(SOURCE_PATH, EVIDENCE_PATH)

    snapshot = corpus.create_snapshot(
        snapshot_id="example-snapshot-001",
        schema_version="evidence-schema-v1",
        source_registry_version="example-sources-v1",
        feature_transformation_version="feature-row-v1",
        created_date="2026-05-24",
    )

    assert snapshot.evidence_count == 6
    assert snapshot.source_count == 1
    assert snapshot.included_source_ids == ("src-illustrative-add-cases",)
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_example_corpus.py -v
```

Expected: FAIL because `data/examples/sources.json` and `data/examples/evidence.jsonl` do not exist.

---

### Task 3: Example Corpus Data

**Files:**
- Create: `data/examples/README.md`
- Create: `data/examples/sources.json`
- Create: `data/examples/evidence.jsonl`

- [ ] **Step 1: Create example corpus documentation**

Create `data/examples/README.md` explaining that the files are synthetic examples, not calibration data, and can be loaded with `core_model.evidence_io.load_corpus`.

- [ ] **Step 2: Create source registry file**

Create `data/examples/sources.json` with one source:

```json
[
  {
    "source_id": "src-illustrative-add-cases",
    "source_name": "Illustrative ADD synthetic cases",
    "source_type": "synthetic",
    "owner_or_publisher": "ADD project",
    "license_or_access": "repository example",
    "update_cadence": "one-time",
    "coverage": ["creative", "education", "finance", "health", "law", "software"],
    "known_biases": ["fictional examples", "not representative", "not calibrated"],
    "quality_tier": "tier_4",
    "status": "experimental"
  }
]
```

- [ ] **Step 3: Create evidence JSONL**

Create `data/examples/evidence.jsonl` with six records whose `source_id` is `src-illustrative-add-cases`, whose `source_quality` is `tier_4`, and whose domains are creative, health, law, finance, software, and education.

- [ ] **Step 4: Run example corpus tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_example_corpus.py -v
```

Expected: all example corpus tests pass.

- [ ] **Step 5: Commit example corpus**

```bash
git add data/examples tests/test_example_corpus.py
git commit -m "test: add illustrative example corpus"
```

---

### Task 4: Public Documentation Updates

**Files:**
- Modify: `README.md`
- Modify: `docs/evidence-data-architecture.md`
- Modify: `docs/validation-agenda.md`

- [ ] **Step 1: Update README**

Add `docs/adjudication-protocol.md` to the Current Foundation list and project structure. Add `data/examples/` to project structure with notes that it contains illustrative example corpus files.

- [ ] **Step 2: Update evidence data architecture**

Reference `docs/adjudication-protocol.md` and `data/examples/` in the implementation path and reference implementation sections. Make clear that the example corpus is synthetic and excluded from calibration by default.

- [ ] **Step 3: Update validation agenda**

In retrospective scoring, state that case labeling should follow `docs/adjudication-protocol.md`, including reviewer disagreement handling and calibration eligibility gates.

- [ ] **Step 4: Run full verification**

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
roots = [Path("README.md"), Path("docs"), Path("core_model"), Path("tests"), Path("data")]
for root in roots:
    paths = [root] if root.is_file() else root.rglob("*")
    for path in paths:
        if path.is_file() and path.suffix in {".md", ".py", ".json", ".jsonl"}:
            text = path.read_text(encoding="utf-8")
            for term in terms:
                if term in text:
                    print(f"{path}: contains {term!r}")
PY
rg -n "\b(t[i]ck|d[e]er|a[c]orn|f[o]rest)\b" README.md docs core_model tests data || true
git status --short --branch
```

Expected:

- pytest passes,
- no prohibited overclaim terms are reported,
- no wrong-domain terms are reported,
- status shows only intended documentation changes before commit.

- [ ] **Step 5: Commit documentation updates**

```bash
git add README.md docs/evidence-data-architecture.md docs/validation-agenda.md
git commit -m "docs: connect adjudication protocol to evidence architecture"
```

---

## Final Verification

- [ ] **Step 1: Run full tests**

```bash
.venv/bin/python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Check example corpus loads manually**

```bash
.venv/bin/python - <<'PY'
from core_model.evidence_io import load_corpus

corpus = load_corpus("data/examples/sources.json", "data/examples/evidence.jsonl")
print(corpus.coverage_summary())
print(corpus.calibration_evidence())
PY
```

Expected: summary includes six domains and calibration evidence is empty.

- [ ] **Step 3: Check status**

```bash
git status --short --branch
```

Expected: working tree is clean on the feature branch before merge.
