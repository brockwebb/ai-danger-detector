# ADD Model Comparison Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a model-comparison layer that evaluates named ADD scorers on the same corpus while separating common TEVV decision metrics from model-native metric compatibility.

**Architecture:** Create `core_model/model_comparison.py` as a focused reporting layer over the existing `evaluate_corpus` API. Each scorer declares output type, native metric notes, deferred metrics, and equivalence notes; the comparison report preserves one ordinal evaluation report per scorer and exposes only explicit metric-by-metric comparison helpers.

**Tech Stack:** Python 3.12-3.14, dataclasses, standard-library enums, existing ADD evidence/evaluation/rubric APIs, pytest.

---

## File Structure

- Create `core_model/model_comparison.py`: scorer metadata, comparison rows/reports, baseline scorer registration, multi-scorer comparison.
- Modify `core_model/__init__.py`: export model-comparison API.
- Create `tests/test_model_comparison.py`: TDD coverage for scorer metadata, comparison reports, metric compatibility, explicit best-by-metric selection, and exports.
- Modify `README.md`: list `model_comparison.py` and `test_model_comparison.py`.
- Modify `docs/evidence-data-architecture.md`: document common decision metrics versus model-native metrics and equivalence notes.
- Modify `docs/validation-agenda.md`: describe model comparison as part of the TEVV loop.

---

### Task 1: Model Comparison Core

**Files:**
- Create: `tests/test_model_comparison.py`
- Create: `core_model/model_comparison.py`
- Modify: `core_model/__init__.py`

- [ ] **Step 1: Write failing tests for scorer metadata and comparison reports**

Create `tests/test_model_comparison.py`:

```python
from core_model.evidence_corpus import EvidenceCorpus
from core_model.evidence_io import load_corpus
from core_model.evidence_schema import (
    EvidenceQualityTier,
    EvidenceType,
    EvidenceUnit,
    OutcomeLabel,
    OversightLabel,
    UserExpertise,
)
from core_model.evaluation_runner import EvaluationReport
from core_model.model_comparison import (
    ScorerDefinition,
    ScorerOutputType,
    baseline_rubric_scorer,
    compare_models,
)
from core_model.rubric_scorer import RubricScore
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


def _score_as(label):
    def scorer(unit):
        return RubricScore(
            score=0.0,
            band=label,
            factor_scores={},
            drivers=("test scorer",),
            assumptions=("test-only scorer",),
        )

    return scorer


def _score_as_adjudicated(unit):
    band = (
        OversightLabel.CASUAL_EXPLORATORY
        if unit.oversight_label is OversightLabel.UNKNOWN
        else unit.oversight_label
    )
    return RubricScore(
        score=1.0,
        band=band,
        factor_scores={},
        drivers=("test oracle",),
        assumptions=("test-only scorer",),
    )


def test_baseline_rubric_scorer_declares_metric_compatibility():
    scorer = baseline_rubric_scorer()

    assert scorer.name == "baseline_rubric"
    assert scorer.output_type is ScorerOutputType.ORDINAL_BAND
    assert scorer.scorer is not None
    assert "brier_score" in scorer.deferred_metric_names
    assert "log_loss" in scorer.deferred_metric_names
    assert "expected_calibration_error" in scorer.deferred_metric_names
    assert any("not a calibrated probability" in note for note in scorer.native_metric_notes)
    assert any("directly" in note for note in scorer.equivalence_notes)


def test_compare_models_evaluates_example_corpus_with_baseline_scorer():
    corpus = load_corpus("data/examples/sources.json", "data/examples/evidence.jsonl")

    report = compare_models(corpus, (baseline_rubric_scorer(),))

    assert report.scorer_count == 1
    assert report.record_count == 6
    assert report.coverage_summary["by_source_id"] == {"src-illustrative-add-cases": 6}
    assert report.rows[0].scorer_name == "baseline_rubric"
    assert isinstance(report.rows[0].evaluation_report, EvaluationReport)
    assert report.rows[0].common_metrics == report.rows[0].evaluation_report.metrics
    assert report.rows[0].common_metrics["exact_band_agreement"] == 0.5


def test_compare_models_selects_best_by_explicit_metric_only():
    corpus = _corpus(
        _evidence("case-001", oversight_label=OversightLabel.CASUAL_EXPLORATORY),
        _evidence("case-002", oversight_label=OversightLabel.EXPERT_REVIEW_REQUIRED),
    )
    always_low = ScorerDefinition(
        name="always_low",
        description="Always predicts the lowest oversight band.",
        output_type=ScorerOutputType.ORDINAL_BAND,
        scorer=_score_as(OversightLabel.CASUAL_EXPLORATORY),
        native_metric_notes=("Ordinal band-error metrics apply.",),
        equivalence_notes=("Returns a band directly for comparison.",),
    )
    oracle = ScorerDefinition(
        name="oracle",
        description="Returns the adjudicated label for test comparison.",
        output_type=ScorerOutputType.ORDINAL_BAND,
        scorer=_score_as_adjudicated,
        native_metric_notes=("Ordinal band-error metrics apply.",),
        equivalence_notes=("Returns a band directly for comparison.",),
    )

    report = compare_models(corpus, (always_low, oracle))

    assert report.scorer_count == 2
    assert report.best_by_metric("exact_band_agreement").scorer_name == "oracle"
    assert (
        report.best_by_metric("mean_absolute_band_error", lower_is_better=True).scorer_name
        == "oracle"
    )


def test_probability_metrics_are_deferred_for_baseline_rubric():
    corpus = load_corpus("data/examples/sources.json", "data/examples/evidence.jsonl")

    report = compare_models(corpus, (baseline_rubric_scorer(),))
    row = report.rows[0]

    assert row.native_metrics == {}
    assert "brier_score" not in row.common_metrics
    assert "log_loss" not in row.common_metrics
    assert "expected_calibration_error" not in row.common_metrics
    assert row.deferred_metric_names == (
        "brier_score",
        "log_loss",
        "expected_calibration_error",
    )


def test_model_comparison_exports_public_api():
    import core_model

    assert "ScorerDefinition" in core_model.__all__
    assert "ScorerOutputType" in core_model.__all__
    assert "ModelComparisonReport" in core_model.__all__
    assert "ModelComparisonRow" in core_model.__all__
    assert "baseline_rubric_scorer" in core_model.__all__
    assert "compare_models" in core_model.__all__
    assert core_model.baseline_rubric_scorer is baseline_rubric_scorer
    assert core_model.compare_models is compare_models
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_model_comparison.py -v
```

Expected: FAIL during import because `core_model.model_comparison` does not exist.

- [ ] **Step 3: Implement model comparison module**

Create `core_model/model_comparison.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .evidence_corpus import EvidenceCorpus
from .evidence_schema import EvidenceUnit, OversightLabel
from .evaluation_runner import EvaluationReport, evaluate_corpus
from .rubric_scorer import RubricScore, score_evidence_unit


class ScorerOutputType(str, Enum):
    ORDINAL_BAND = "ordinal_band"
    PROBABILITY = "probability"
    DISTRIBUTION = "distribution"
    WORKFLOW = "workflow"
    ENSEMBLE = "ensemble"


@dataclass(frozen=True)
class ScorerDefinition:
    name: str
    description: str
    output_type: ScorerOutputType
    scorer: Callable[[EvidenceUnit], RubricScore]
    native_metric_notes: tuple[str, ...] = ()
    equivalence_notes: tuple[str, ...] = ()
    deferred_metric_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelComparisonRow:
    scorer_name: str
    description: str
    output_type: ScorerOutputType
    evaluation_report: EvaluationReport
    common_metrics: dict[str, float]
    native_metrics: dict[str, float]
    native_metric_notes: tuple[str, ...]
    equivalence_notes: tuple[str, ...]
    deferred_metric_names: tuple[str, ...]


@dataclass(frozen=True)
class ModelComparisonReport:
    rows: tuple[ModelComparisonRow, ...]
    record_count: int
    coverage_summary: dict[str, dict[str, int]]

    @property
    def scorer_count(self) -> int:
        return len(self.rows)

    def best_by_metric(
        self, metric_name: str, *, lower_is_better: bool = False
    ) -> ModelComparisonRow:
        candidates = tuple(row for row in self.rows if metric_name in row.common_metrics)
        if not candidates:
            raise KeyError(f"metric not available for comparison: {metric_name}")
        key = lambda row: row.common_metrics[metric_name]
        return min(candidates, key=key) if lower_is_better else max(candidates, key=key)


def baseline_rubric_scorer() -> ScorerDefinition:
    return ScorerDefinition(
        name="baseline_rubric",
        description="Provisional ordinal rubric scorer for oversight-band triage.",
        output_type=ScorerOutputType.ORDINAL_BAND,
        scorer=score_evidence_unit,
        native_metric_notes=(
            "Ordinal band-error metrics apply.",
            "Threshold decision metrics apply after mapping bands to escalation thresholds.",
            "Brier score, log loss, and expected calibration error are deferred because the score is not a calibrated probability.",
        ),
        equivalence_notes=(
            "The rubric returns an oversight band directly; no probability-to-band threshold was applied.",
        ),
        deferred_metric_names=(
            "brier_score",
            "log_loss",
            "expected_calibration_error",
        ),
    )


def _comparison_row(
    corpus: EvidenceCorpus,
    scorer: ScorerDefinition,
    threshold_band: OversightLabel,
) -> ModelComparisonRow:
    evaluation_report = evaluate_corpus(
        corpus,
        scorer=scorer.scorer,
        threshold_band=threshold_band,
    )
    return ModelComparisonRow(
        scorer_name=scorer.name,
        description=scorer.description,
        output_type=scorer.output_type,
        evaluation_report=evaluation_report,
        common_metrics=dict(evaluation_report.metrics),
        native_metrics={},
        native_metric_notes=scorer.native_metric_notes,
        equivalence_notes=scorer.equivalence_notes,
        deferred_metric_names=scorer.deferred_metric_names,
    )


def compare_models(
    corpus: EvidenceCorpus,
    scorers: tuple[ScorerDefinition, ...],
    *,
    threshold_band: OversightLabel = OversightLabel.TRAINED_REVIEW_REQUIRED,
) -> ModelComparisonReport:
    if not scorers:
        raise ValueError("at least one scorer is required")
    rows = tuple(_comparison_row(corpus, scorer, threshold_band) for scorer in scorers)
    return ModelComparisonReport(
        rows=rows,
        record_count=len(corpus.evidence),
        coverage_summary=corpus.coverage_summary(),
    )
```

- [ ] **Step 4: Export model comparison API**

Update `core_model/__init__.py` imports:

```python
from .model_comparison import (
    ModelComparisonReport,
    ModelComparisonRow,
    ScorerDefinition,
    ScorerOutputType,
    baseline_rubric_scorer,
    compare_models,
)
```

Add to `__all__`:

```python
"ModelComparisonReport",
"ModelComparisonRow",
"ScorerDefinition",
"ScorerOutputType",
"baseline_rubric_scorer",
"compare_models",
```

- [ ] **Step 5: Run tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_model_comparison.py -v
```

Expected: all model-comparison tests pass.

- [ ] **Step 6: Run related tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_model_comparison.py tests/test_evaluation_runner.py tests/test_rubric_scorer.py tests/test_example_corpus.py -v
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit core implementation**

```bash
git add core_model/model_comparison.py core_model/__init__.py tests/test_model_comparison.py
git commit -m "feat: compare named scorers with metric compatibility"
```

---

### Task 2: Documentation and Metric Compatibility Notes

**Files:**
- Modify: `README.md`
- Modify: `docs/evidence-data-architecture.md`
- Modify: `docs/validation-agenda.md`

- [ ] **Step 1: Update README project structure**

In `README.md`, update the current implementation paragraph to include model comparison:

```markdown
The active Python implementation lives in `core_model/`. It includes the numerical framework plus an evidence backbone for typed evidence records, source registry metadata, corpus management, local data loading, reproducible data snapshots, a provisional rubric scorer, an ordinal evaluation runner, a model-comparison layer, and calibration-oriented performance metrics.
```

Add to the `core_model/` tree:

```markdown
|   |-- model_comparison.py        # Named scorer comparison and metric compatibility
```

Add to the `tests/` tree:

```markdown
|   |-- test_model_comparison.py   # Model comparison and metric compatibility tests
```

- [ ] **Step 2: Update evidence architecture metric-layer section**

In `docs/evidence-data-architecture.md`, after the paragraph describing `core_model/evaluation_runner.py`, add:

```markdown
The model-comparison layer in `core_model/model_comparison.py` separates common decision-layer metrics from model-native metrics. Common metrics compare scorers after they produce or are mapped to oversight bands. Model-native metrics are reported only when the scorer output supports them: ordinal scorers get band-error metrics, calibrated probability scorers can get Brier score or log loss, Bayesian scorers can report posterior or posterior-predictive summaries, and Markov workflow models can report workflow path or transition-fit diagnostics when evidence supports those claims. Each comparison row carries equivalence notes so readers can see how a model output was translated into the shared oversight-band decision target.
```

In the implementation path list, add:

```markdown
- `core_model/model_comparison.py` for named scorer comparison, metric compatibility notes, and TEVV decision-layer summaries.
```

In the test list, add:

```markdown
- `tests/test_model_comparison.py` for named scorer comparison and metric-compatibility behavior.
```

In the reference implementation paragraph, add `core_model/model_comparison.py` alongside the other active backend modules.

- [ ] **Step 3: Update validation agenda TEVV expectations**

In `docs/validation-agenda.md`, after the retrospective scoring paragraph, add:

```markdown
Model comparison should be part of the TEVV loop. Each candidate model should be evaluated against the same case set where possible, but comparison reports should not force incompatible outputs into fake equivalence. Shared oversight-band decisions can be compared with ordinal and threshold metrics. Model-native metrics should be reported only when the output type supports them; for example, Brier score and log loss require calibrated probability outputs, while Markov workflow models need workflow-transition evidence before path-fit claims are meaningful.
```

- [ ] **Step 4: Run full verification and scans**

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
git commit -m "docs: document model comparison metrics"
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
names = {
    "ScorerDefinition",
    "ScorerOutputType",
    "ModelComparisonReport",
    "ModelComparisonRow",
    "baseline_rubric_scorer",
    "compare_models",
}
print({name: name in core_model.__all__ for name in sorted(names)})
PY
```

Expected: every printed value is `True`.

- [ ] **Step 3: Manually compare the example corpus**

```bash
.venv/bin/python - <<'PY'
from core_model.evidence_io import load_corpus
from core_model.model_comparison import baseline_rubric_scorer, compare_models

corpus = load_corpus("data/examples/sources.json", "data/examples/evidence.jsonl")
report = compare_models(corpus, (baseline_rubric_scorer(),))
row = report.rows[0]
print(report.scorer_count)
print(row.scorer_name)
print(row.common_metrics)
print(row.deferred_metric_names)
PY
```

Expected: output shows one scorer named `baseline_rubric`, ordinal common metrics, and deferred probability metric names.

- [ ] **Step 4: Check status**

```bash
git status --short --branch
```

Expected: working tree is clean on the feature branch before merge.

## Success Checklist

- `core_model/model_comparison.py` exists and compares named scorers.
- The baseline rubric scorer declares metric compatibility and deferred probability metrics.
- `compare_models` returns one row per scorer with a full `EvaluationReport`.
- `best_by_metric` compares only the metric the caller names.
- Documentation separates common decision-layer metrics from model-native metrics.
- Full test suite passes.
