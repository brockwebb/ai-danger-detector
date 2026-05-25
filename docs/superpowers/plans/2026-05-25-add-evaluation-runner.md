# ADD Evaluation Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an ordinal evaluation runner that scores evidence corpora and summarizes baseline scorer performance against adjudicated oversight labels.

**Architecture:** Create `core_model/evaluation_runner.py` as a focused reporting layer over `EvidenceCorpus` and `score_evidence_unit`. The runner creates traceable per-record rows and aggregate ordinal metrics; it does not compute probability calibration metrics because the current rubric score is not calibrated.

**Tech Stack:** Python 3.12-3.14, dataclasses, standard-library collections, pytest, existing ADD corpus/scorer APIs.

---

## File Structure

- Create `core_model/evaluation_runner.py`: row/report dataclasses, ordinal band mapping, corpus evaluation.
- Modify `core_model/__init__.py`: export evaluation runner API.
- Create `tests/test_evaluation_runner.py`: behavior tests for rows, metrics, unknown labels, and example corpus evaluation.
- Modify `README.md`: include `evaluation_runner.py` and tests in project structure.
- Modify `docs/evidence-data-architecture.md`: document the runner as the first scoring evaluation path.
- Modify `docs/validation-agenda.md`: reference runner outputs in retrospective scoring and reporting.

---

### Task 1: Evaluation Rows and Basic Metrics

**Files:**
- Create: `tests/test_evaluation_runner.py`
- Create: `core_model/evaluation_runner.py`

- [ ] **Step 1: Write failing tests for row traceability and ordinal metrics**

Create `tests/test_evaluation_runner.py` with helpers for active source registry and evidence records. Add tests:

```python
from core_model.evaluation_runner import evaluate_corpus
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
        source_name="Active adjudicated cases",
        source_type=SourceType.CASE_SET,
        owner_or_publisher="ADD",
        license_or_access="private",
        update_cadence="one-time",
        coverage=("creative", "health", "law"),
        known_biases=(),
        quality_tier=EvidenceQualityTier.TIER_1,
    )
    registry.update_status("src-active", SourceStatus.ACTIVE, reason="approved")
    return registry


def _evidence(evidence_id, *, oversight_label, **overrides):
    values = {
        "evidence_id": evidence_id,
        "source_id": "src-active",
        "evidence_type": EvidenceType.CASE_REVIEW,
        "collection_date": "2026-05-25",
        "event_date": "2026-05-20",
        "domain": "creative",
        "task_type": "draft low-stakes text",
        "model_family": "unknown",
        "model_version": "unknown",
        "user_expertise": UserExpertise.TRAINED,
        "governance_context": "ordinary user review",
        "outcome_label": OutcomeLabel.BENIGN_USE,
        "oversight_label": oversight_label,
        "harm_severity": 0.05,
        "detectability": 0.9,
        "reversibility": 0.95,
        "verification_burden": 0.1,
        "workflow_path": ("S0", "S1", "S3", "S7"),
        "confidence": 0.7,
        "source_quality": EvidenceQualityTier.TIER_1,
        "bias_notes": (),
        "relevance_limits": (),
    }
    values.update(overrides)
    return EvidenceUnit(**values)


def _corpus(*units):
    corpus = EvidenceCorpus(_registry())
    for unit in units:
        corpus.add(unit)
    return corpus


def test_evaluate_corpus_preserves_traceable_rows():
    corpus = _corpus(_evidence("case-001", oversight_label=OversightLabel.CASUAL_EXPLORATORY))

    report = evaluate_corpus(corpus)

    assert report.record_count == 1
    assert report.rows[0].evidence_id == "case-001"
    assert report.rows[0].source_id == "src-active"
    assert report.rows[0].domain == "creative"
    assert report.rows[0].adjudicated_band is OversightLabel.CASUAL_EXPLORATORY
    assert report.rows[0].predicted_band is OversightLabel.CASUAL_EXPLORATORY
    assert report.rows[0].error_direction == "match"
    assert isinstance(report.rows[0].drivers, tuple)


def test_evaluate_corpus_reports_agreement_and_band_error():
    corpus = _corpus(
        _evidence("case-001", oversight_label=OversightLabel.CASUAL_EXPLORATORY),
        _evidence(
            "case-002",
            oversight_label=OversightLabel.EXPERT_REVIEW_REQUIRED,
            domain="health",
            harm_severity=0.9,
            detectability=0.2,
            reversibility=0.2,
            verification_burden=0.9,
            user_expertise=UserExpertise.NON_EXPERT,
            governance_context="no expert review",
        ),
    )

    report = evaluate_corpus(corpus)

    assert report.record_count == 2
    assert report.evaluable_count == 2
    assert report.metrics["exact_band_agreement"] == 0.5
    assert report.metrics["mean_absolute_band_error"] == 0.5
    assert report.predicted_band_counts["casual_exploratory"] == 1
    assert report.adjudicated_band_counts["expert_review_required"] == 1
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_runner.py -v
```

Expected: FAIL because `core_model.evaluation_runner` does not exist.

- [ ] **Step 3: Implement row/report dataclasses and basic evaluation**

Create `core_model/evaluation_runner.py` with:

```python
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

from .evidence_corpus import EvidenceCorpus
from .evidence_schema import EvidenceUnit, OversightLabel
from .rubric_scorer import RubricScore, score_evidence_unit


BAND_ORDINALS = {
    OversightLabel.CASUAL_EXPLORATORY: 1,
    OversightLabel.ASSISTED_BOUNDED: 2,
    OversightLabel.TRAINED_REVIEW_REQUIRED: 3,
    OversightLabel.EXPERT_REVIEW_REQUIRED: 4,
    OversightLabel.EXPERT_LED_OR_NO_AUTONOMOUS_USE: 5,
}
```

Add:

- `EvaluationRow` dataclass,
- `EvaluationReport` dataclass,
- `_band_error_direction`,
- `_safe_rate`,
- `evaluate_corpus`.

Initial implementation should compute row details, exact agreement, mean absolute band error, under-escalation rate, over-escalation rate, false reassurance rate, false escalation rate, predicted/adjudicated band counts, and coverage summary.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_runner.py -v
```

Expected: current evaluation runner tests pass.

- [ ] **Step 5: Commit basic runner**

```bash
git add core_model/evaluation_runner.py tests/test_evaluation_runner.py
git commit -m "feat: add ordinal evaluation runner"
```

---

### Task 2: Under/Over Escalation, Unknown Labels, Example Corpus, and Exports

**Files:**
- Modify: `tests/test_evaluation_runner.py`
- Modify: `core_model/evaluation_runner.py`
- Modify: `core_model/__init__.py`

- [ ] **Step 1: Add failing tests for under/over escalation, unknown labels, and example corpus**

Append to `tests/test_evaluation_runner.py`:

```python
from core_model.evidence_io import load_corpus
from core_model.evaluation_runner import EvaluationReport, EvaluationRow


def test_evaluate_corpus_counts_under_and_over_escalation():
    corpus = _corpus(
        _evidence(
            "case-under",
            oversight_label=OversightLabel.EXPERT_LED_OR_NO_AUTONOMOUS_USE,
            harm_severity=0.2,
            detectability=0.9,
            reversibility=0.9,
            verification_burden=0.2,
        ),
        _evidence(
            "case-over",
            oversight_label=OversightLabel.ASSISTED_BOUNDED,
            harm_severity=0.95,
            detectability=0.2,
            reversibility=0.2,
            verification_burden=0.95,
            user_expertise=UserExpertise.NON_EXPERT,
            governance_context="no review",
        ),
    )

    report = evaluate_corpus(corpus)

    assert report.metrics["under_escalation_rate"] == 0.5
    assert report.metrics["over_escalation_rate"] == 0.5
    assert report.metrics["false_reassurance_rate"] == 0.5
    assert report.metrics["false_escalation_rate"] == 0.0


def test_evaluate_corpus_retains_unknown_label_rows_but_excludes_from_metrics():
    corpus = _corpus(
        _evidence("case-known", oversight_label=OversightLabel.CASUAL_EXPLORATORY),
        _evidence("case-unknown", oversight_label=OversightLabel.UNKNOWN),
    )

    report = evaluate_corpus(corpus)

    assert report.record_count == 2
    assert report.evaluable_count == 1
    assert report.rows[1].is_evaluable is False
    assert report.rows[1].error_direction == "not_evaluable"
    assert report.metrics["exact_band_agreement"] == 1.0


def test_example_corpus_can_be_evaluated_end_to_end():
    corpus = load_corpus("data/examples/sources.json", "data/examples/evidence.jsonl")

    report = evaluate_corpus(corpus)

    assert isinstance(report, EvaluationReport)
    assert all(isinstance(row, EvaluationRow) for row in report.rows)
    assert report.record_count == 6
    assert report.evaluable_count == 6
    assert report.coverage_summary["by_source_id"] == {"src-illustrative-add-cases": 6}
    assert report.predicted_band_counts["expert_led_or_no_autonomous_use"] >= 2
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_runner.py -v
```

Expected: FAIL if exports, unknown-label handling, or metrics are incomplete.

- [ ] **Step 3: Complete evaluation runner and exports**

Update `core_model/evaluation_runner.py` to handle unknown adjudicated labels as not evaluable and ensure all metric keys are always present.

Update `core_model/__init__.py`:

```python
from .evaluation_runner import (
    BAND_ORDINALS,
    EvaluationReport,
    EvaluationRow,
    evaluate_corpus,
)
```

Add names to `__all__`.

- [ ] **Step 4: Run evaluation tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_runner.py -v
```

Expected: all evaluation runner tests pass.

- [ ] **Step 5: Run related tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_runner.py tests/test_rubric_scorer.py tests/test_example_corpus.py -v
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit evaluation integration**

```bash
git add core_model tests
git commit -m "feat: evaluate example corpus with baseline scorer"
```

---

### Task 3: Documentation and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/evidence-data-architecture.md`
- Modify: `docs/validation-agenda.md`

- [ ] **Step 1: Update README**

Add `evaluation_runner.py` and `test_evaluation_runner.py` to the project structure.

- [ ] **Step 2: Update evidence data architecture**

Document the evaluation runner as the first ordinal evaluation path for comparing predicted oversight bands with adjudicated labels.

- [ ] **Step 3: Update validation agenda**

Mention exact band agreement, mean absolute band error, under-escalation, over-escalation, false reassurance, and false escalation as first-run retrospective scoring outputs.

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
git commit -m "docs: document ordinal evaluation runner"
```

---

## Final Verification

- [ ] **Step 1: Run full tests**

```bash
.venv/bin/python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Check exports**

```bash
.venv/bin/python - <<'PY'
import core_model
print(core_model.__all__)
PY
```

Expected: exports include `EvaluationReport`, `EvaluationRow`, and `evaluate_corpus`.

- [ ] **Step 3: Evaluate example corpus manually**

```bash
.venv/bin/python - <<'PY'
from core_model.evidence_io import load_corpus
from core_model.evaluation_runner import evaluate_corpus

corpus = load_corpus("data/examples/sources.json", "data/examples/evidence.jsonl")
report = evaluate_corpus(corpus)
print(report.metrics)
print(report.predicted_band_counts)
PY
```

Expected: metrics and predicted band counts are printed.

- [ ] **Step 4: Check status**

```bash
git status --short --branch
```

Expected: working tree is clean on the feature branch before merge.
