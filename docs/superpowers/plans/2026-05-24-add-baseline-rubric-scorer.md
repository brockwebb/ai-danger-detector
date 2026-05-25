# ADD Baseline Rubric Scorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a transparent baseline rubric scorer that turns ADD evidence units and feature rows into provisional oversight scores and bands.

**Architecture:** Create `core_model/rubric_scorer.py` as a focused scoring module with dataclass configuration and result objects. The scorer uses existing evidence fields, explicit weights, ordinal band thresholds, and conservative escalation rules; it does not train, calibrate, or claim empirical validity.

**Tech Stack:** Python 3.12-3.14, dataclasses, pytest, existing ADD evidence schema/corpus/loader.

---

## File Structure

- Create `core_model/rubric_scorer.py`: scoring config, score result, evidence-unit scoring, feature-row scoring.
- Modify `core_model/__init__.py`: export scorer API.
- Create `tests/test_rubric_scorer.py`: behavior tests for bands, monotonicity, feature rows, and example corpus scoring.
- Modify `README.md`: include `rubric_scorer.py` and tests in project structure.
- Modify `docs/model-rubric.md`: document the reference scorer and its provisional status.
- Modify `docs/evidence-data-architecture.md`: list the scorer as the first baseline model consuming the evidence layer.

---

### Task 1: Core Scorer API

**Files:**
- Create: `tests/test_rubric_scorer.py`
- Create: `core_model/rubric_scorer.py`

- [ ] **Step 1: Write failing tests for scoring output and monotonicity**

Create `tests/test_rubric_scorer.py` with helper evidence and these tests:

```python
from dataclasses import replace

from core_model.evidence_schema import (
    EvidenceQualityTier,
    EvidenceType,
    EvidenceUnit,
    OutcomeLabel,
    OversightLabel,
    UserExpertise,
)
from core_model.rubric_scorer import score_evidence_unit


def _evidence(**overrides):
    values = {
        "evidence_id": "case-001",
        "source_id": "src-active",
        "evidence_type": EvidenceType.CASE_REVIEW,
        "collection_date": "2026-05-24",
        "event_date": "2026-05-20",
        "domain": "creative",
        "task_type": "draft low-stakes text",
        "model_family": "unknown",
        "model_version": "unknown",
        "user_expertise": UserExpertise.TRAINED,
        "governance_context": "ordinary user review",
        "outcome_label": OutcomeLabel.BENIGN_USE,
        "oversight_label": OversightLabel.CASUAL_EXPLORATORY,
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


def test_low_consequence_case_scores_to_casual_exploratory():
    result = score_evidence_unit(_evidence())

    assert 0.0 <= result.score < 0.2
    assert result.band is OversightLabel.CASUAL_EXPLORATORY
    assert "provisional" in " ".join(result.assumptions)


def test_increasing_harm_increases_score():
    low = score_evidence_unit(_evidence(harm_severity=0.1))
    high = score_evidence_unit(_evidence(harm_severity=0.9))

    assert high.score > low.score


def test_lower_detectability_increases_score():
    easy = score_evidence_unit(_evidence(detectability=0.9))
    hard = score_evidence_unit(_evidence(detectability=0.1))

    assert hard.score > easy.score


def test_lower_source_quality_adds_uncertainty_driver():
    strong = score_evidence_unit(_evidence(source_quality=EvidenceQualityTier.TIER_1))
    weak = score_evidence_unit(_evidence(source_quality=EvidenceQualityTier.TIER_4))

    assert weak.score > strong.score
    assert "source uncertainty" in weak.drivers
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_rubric_scorer.py -v
```

Expected: FAIL because `core_model.rubric_scorer` does not exist.

- [ ] **Step 3: Implement minimal scorer API**

Create `core_model/rubric_scorer.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any

from .evidence_schema import EvidenceQualityTier, EvidenceUnit, OversightLabel, UserExpertise


@dataclass(frozen=True)
class RubricScoringConfig:
    weights: dict[str, float]
    thresholds: tuple[tuple[float, OversightLabel], ...]


@dataclass(frozen=True)
class RubricScore:
    score: float
    band: OversightLabel
    factor_scores: dict[str, float]
    drivers: tuple[str, ...]
    assumptions: tuple[str, ...]


DEFAULT_RUBRIC_CONFIG = RubricScoringConfig(
    weights={
        "harm_severity": 0.24,
        "low_detectability": 0.16,
        "low_reversibility": 0.14,
        "verification_burden": 0.18,
        "user_expertise_risk": 0.12,
        "governance_risk": 0.10,
        "source_uncertainty_risk": 0.06,
    },
    thresholds=(
        (0.80, OversightLabel.EXPERT_LED_OR_NO_AUTONOMOUS_USE),
        (0.60, OversightLabel.EXPERT_REVIEW_REQUIRED),
        (0.40, OversightLabel.TRAINED_REVIEW_REQUIRED),
        (0.20, OversightLabel.ASSISTED_BOUNDED),
        (0.00, OversightLabel.CASUAL_EXPLORATORY),
    ),
)
```

Implement helper functions `_risk_from_user_expertise`, `_risk_from_governance_context`, `_risk_from_source_quality`, `_band_for_score`, `_drivers`, and `score_evidence_unit`.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_rubric_scorer.py -v
```

Expected: current rubric scorer tests pass.

- [ ] **Step 5: Commit core scorer**

```bash
git add core_model/rubric_scorer.py tests/test_rubric_scorer.py
git commit -m "feat: add baseline rubric scorer"
```

---

### Task 2: Escalation, Feature Rows, Example Corpus, and Exports

**Files:**
- Modify: `tests/test_rubric_scorer.py`
- Modify: `core_model/rubric_scorer.py`
- Modify: `core_model/__init__.py`

- [ ] **Step 1: Add failing tests for escalation rules, feature rows, and example corpus scoring**

Append to `tests/test_rubric_scorer.py`:

```python
from core_model.evidence_corpus import EvidenceCorpus
from core_model.evidence_io import load_corpus
from core_model.source_registry import SourceRegistry, SourceStatus, SourceType
from core_model.rubric_scorer import score_feature_row


def test_high_harm_low_detectability_escalates_to_expert_review():
    result = score_evidence_unit(
        _evidence(
            domain="health",
            harm_severity=0.85,
            detectability=0.2,
            reversibility=0.5,
            verification_burden=0.8,
            user_expertise=UserExpertise.NON_EXPERT,
            governance_context="no expert review",
        )
    )

    assert result.band in {
        OversightLabel.EXPERT_REVIEW_REQUIRED,
        OversightLabel.EXPERT_LED_OR_NO_AUTONOMOUS_USE,
    }
    assert any("high harm" in driver for driver in result.drivers)


def test_extreme_harm_low_detection_and_low_reversibility_escalates_to_expert_led():
    result = score_evidence_unit(
        _evidence(
            harm_severity=0.95,
            detectability=0.2,
            reversibility=0.2,
            verification_burden=0.95,
            user_expertise=UserExpertise.NON_EXPERT,
            governance_context="no review",
        )
    )

    assert result.band is OversightLabel.EXPERT_LED_OR_NO_AUTONOMOUS_USE


def test_feature_rows_can_be_scored():
    registry = SourceRegistry()
    registry.add_source(
        source_id="src-active",
        source_name="Active cases",
        source_type=SourceType.CASE_SET,
        owner_or_publisher="ADD",
        license_or_access="private",
        update_cadence="one-time",
        coverage=("creative",),
        known_biases=(),
        quality_tier=EvidenceQualityTier.TIER_1,
    )
    registry.update_status("src-active", SourceStatus.ACTIVE, reason="approved")
    corpus = EvidenceCorpus(registry)
    corpus.add(_evidence())

    result = score_feature_row(corpus.feature_rows()[0])

    assert result.band is OversightLabel.CASUAL_EXPLORATORY
    assert result.factor_scores["harm_severity"] == 0.05


def test_example_corpus_scores_high_stakes_above_creative_case():
    corpus = load_corpus("data/examples/sources.json", "data/examples/evidence.jsonl")

    scores = {
        unit.domain: score_evidence_unit(unit)
        for unit in corpus.evidence
    }

    assert scores["health"].score > scores["creative"].score
    assert scores["finance"].score > scores["education"].score
    assert scores["health"].band is OversightLabel.EXPERT_LED_OR_NO_AUTONOMOUS_USE
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_rubric_scorer.py -v
```

Expected: FAIL because `score_feature_row` is not implemented or escalation behavior is incomplete.

- [ ] **Step 3: Implement escalation rules and feature-row scoring**

Update `core_model/rubric_scorer.py` to:

- apply the conservative escalation rules,
- add escalation drivers,
- implement `score_feature_row(row, config=DEFAULT_RUBRIC_CONFIG)`,
- parse string enum values from feature rows.

Update `core_model/__init__.py`:

```python
from .rubric_scorer import (
    DEFAULT_RUBRIC_CONFIG,
    RubricScore,
    RubricScoringConfig,
    score_evidence_unit,
    score_feature_row,
)
```

Add these names to `__all__`.

- [ ] **Step 4: Run rubric scorer tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_rubric_scorer.py -v
```

Expected: all rubric scorer tests pass.

- [ ] **Step 5: Run related tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_rubric_scorer.py tests/test_example_corpus.py tests/test_evidence_corpus.py -v
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit scorer integration**

```bash
git add core_model tests
git commit -m "feat: score feature rows and example corpus"
```

---

### Task 3: Documentation and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/model-rubric.md`
- Modify: `docs/evidence-data-architecture.md`

- [ ] **Step 1: Update README**

Add `rubric_scorer.py` and `test_rubric_scorer.py` to the project structure.

- [ ] **Step 2: Update model rubric document**

Add a "Reference Scorer" section explaining that `core_model/rubric_scorer.py` is a provisional, assumption-driven baseline that outputs a score, oversight band, factor scores, drivers, and assumptions. State that it is not calibrated.

- [ ] **Step 3: Update evidence data architecture**

Mention that the first baseline scorer consumes `EvidenceUnit` objects and feature rows, making evaluation possible before later Bayesian or ensemble models exist.

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
git add README.md docs/model-rubric.md docs/evidence-data-architecture.md
git commit -m "docs: document baseline rubric scorer"
```

---

## Final Verification

- [ ] **Step 1: Run full tests**

```bash
.venv/bin/python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Check scorer exports**

```bash
.venv/bin/python - <<'PY'
import core_model
print(core_model.__all__)
PY
```

Expected: exports include `RubricScore`, `RubricScoringConfig`, `score_evidence_unit`, and `score_feature_row`.

- [ ] **Step 3: Score example corpus manually**

```bash
.venv/bin/python - <<'PY'
from core_model.evidence_io import load_corpus
from core_model.rubric_scorer import score_evidence_unit

corpus = load_corpus("data/examples/sources.json", "data/examples/evidence.jsonl")
for unit in corpus.evidence:
    result = score_evidence_unit(unit)
    print(unit.evidence_id, result.band.value, round(result.score, 3))
PY
```

Expected: six scored rows are printed.

- [ ] **Step 4: Check status**

```bash
git status --short --branch
```

Expected: working tree is clean on the feature branch before merge.
