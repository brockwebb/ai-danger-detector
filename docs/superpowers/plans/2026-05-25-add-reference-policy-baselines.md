# ADD Reference Policy Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic reference policy scorers so ADD can compare candidate models against simple TEVV controls.

**Architecture:** Create `core_model/reference_scorers.py` as a small registry of fixed-band `ScorerDefinition` factories. The module should reuse `RubricScore`, `ScorerOutputType`, and `compare_models` without changing the comparison layer; these scorers are policy baselines, not learned models.

**Tech Stack:** Python 3.12-3.14, dataclasses already in the codebase, existing ADD scorer/comparison APIs, pytest.

---

## File Structure

- Create `core_model/reference_scorers.py`: fixed-band reference policy scorers and stable ordered registry.
- Modify `core_model/__init__.py`: export the reference scorer factories.
- Create `tests/test_reference_scorers.py`: TDD coverage for fixed bands, metadata, comparison integration, tradeoff metrics, and exports.
- Modify `README.md`: list `reference_scorers.py` and `test_reference_scorers.py`.
- Modify `docs/evidence-data-architecture.md`: document reference policies as TEVV controls.
- Modify `docs/validation-agenda.md`: require candidate models to beat or explain tradeoffs against trivial fixed policies.

---

### Task 1: Reference Policy Scorers

**Files:**
- Create: `tests/test_reference_scorers.py`
- Create: `core_model/reference_scorers.py`
- Modify: `core_model/__init__.py`

- [ ] **Step 1: Write failing tests for fixed policy scorers**

Create `tests/test_reference_scorers.py`:

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
from core_model.model_comparison import ScorerOutputType, baseline_rubric_scorer, compare_models
from core_model.reference_scorers import (
    always_assisted_scorer,
    always_casual_scorer,
    always_expert_led_scorer,
    always_expert_review_scorer,
    always_trained_review_scorer,
    reference_policy_scorers,
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


def test_fixed_policy_scorers_return_expected_bands():
    unit = _evidence("case-001", oversight_label=OversightLabel.CASUAL_EXPLORATORY)

    assert always_casual_scorer().scorer(unit).band is OversightLabel.CASUAL_EXPLORATORY
    assert always_assisted_scorer().scorer(unit).band is OversightLabel.ASSISTED_BOUNDED
    assert (
        always_trained_review_scorer().scorer(unit).band
        is OversightLabel.TRAINED_REVIEW_REQUIRED
    )
    assert (
        always_expert_review_scorer().scorer(unit).band
        is OversightLabel.EXPERT_REVIEW_REQUIRED
    )
    assert (
        always_expert_led_scorer().scorer(unit).band
        is OversightLabel.EXPERT_LED_OR_NO_AUTONOMOUS_USE
    )


def test_reference_policy_scorers_have_stable_order_and_metadata():
    scorers = reference_policy_scorers()

    assert [scorer.name for scorer in scorers] == [
        "always_casual",
        "always_assisted",
        "always_trained_review",
        "always_expert_review",
        "always_expert_led",
    ]
    assert all(scorer.output_type is ScorerOutputType.ORDINAL_BAND for scorer in scorers)
    assert all("policy baseline" in scorer.description for scorer in scorers)
    assert all("brier_score" in scorer.deferred_metric_names for scorer in scorers)
    assert all(any("TEVV" in note for note in scorer.native_metric_notes) for scorer in scorers)
    assert all(any("not learned" in note for note in scorer.equivalence_notes) for scorer in scorers)


def test_reference_policies_run_through_model_comparison():
    corpus = load_corpus("data/examples/sources.json", "data/examples/evidence.jsonl")

    report = compare_models(
        corpus,
        (baseline_rubric_scorer(), *reference_policy_scorers()),
    )

    assert report.scorer_count == 6
    assert [row.scorer_name for row in report.rows][0] == "baseline_rubric"
    assert [row.scorer_name for row in report.rows][1:] == [
        "always_casual",
        "always_assisted",
        "always_trained_review",
        "always_expert_review",
        "always_expert_led",
    ]
    assert all(row.common_metrics for row in report.rows)


def test_reference_policies_expose_expected_tradeoffs():
    corpus = _corpus(
        _evidence("case-low", oversight_label=OversightLabel.CASUAL_EXPLORATORY),
        _evidence(
            "case-high",
            oversight_label=OversightLabel.EXPERT_LED_OR_NO_AUTONOMOUS_USE,
        ),
    )

    report = compare_models(
        corpus,
        (always_casual_scorer(), always_expert_led_scorer()),
    )
    casual_row = report.rows[0]
    expert_led_row = report.rows[1]

    assert casual_row.common_metrics["under_escalation_rate"] == 0.5
    assert casual_row.common_metrics["false_reassurance_rate"] == 1.0
    assert casual_row.common_metrics["over_escalation_rate"] == 0.0
    assert expert_led_row.common_metrics["over_escalation_rate"] == 0.5
    assert expert_led_row.common_metrics["false_escalation_rate"] == 1.0
    assert expert_led_row.common_metrics["under_escalation_rate"] == 0.0


def test_reference_scorers_export_public_api():
    import core_model

    assert "always_casual_scorer" in core_model.__all__
    assert "always_assisted_scorer" in core_model.__all__
    assert "always_trained_review_scorer" in core_model.__all__
    assert "always_expert_review_scorer" in core_model.__all__
    assert "always_expert_led_scorer" in core_model.__all__
    assert "reference_policy_scorers" in core_model.__all__
    assert core_model.always_casual_scorer is always_casual_scorer
    assert core_model.reference_policy_scorers is reference_policy_scorers
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_reference_scorers.py -v
```

Expected: FAIL during import because `core_model.reference_scorers` does not exist.

- [ ] **Step 3: Implement reference scorer module**

Create `core_model/reference_scorers.py`:

```python
from __future__ import annotations

from .evidence_schema import EvidenceUnit, OversightLabel
from .model_comparison import ScorerDefinition, ScorerOutputType
from .rubric_scorer import RubricScore


_DEFERRED_PROBABILITY_METRICS = (
    "brier_score",
    "log_loss",
    "expected_calibration_error",
)


def _fixed_policy_score(band: OversightLabel, name: str) -> RubricScore:
    return RubricScore(
        score=0.0,
        band=band,
        factor_scores={},
        drivers=(f"fixed policy baseline: {name}",),
        assumptions=(
            "Deterministic TEVV policy baseline, not a learned risk model.",
            "Does not emit calibrated probabilities.",
        ),
    )


def _fixed_policy_scorer(
    *,
    name: str,
    description: str,
    band: OversightLabel,
) -> ScorerDefinition:
    def scorer(unit: EvidenceUnit) -> RubricScore:
        return _fixed_policy_score(band, name)

    return ScorerDefinition(
        name=name,
        description=description,
        output_type=ScorerOutputType.ORDINAL_BAND,
        scorer=scorer,
        native_metric_notes=(
            "TEVV control baseline for ordinal and threshold decision metrics.",
            "Useful for exposing false reassurance and over-escalation tradeoffs.",
            "Brier score, log loss, and expected calibration error are deferred because this policy does not emit calibrated probabilities.",
        ),
        equivalence_notes=(
            "The policy returns a fixed oversight band directly.",
            "This is not learned from evidence and should not be treated as a risk model.",
        ),
        deferred_metric_names=_DEFERRED_PROBABILITY_METRICS,
    )


def always_casual_scorer() -> ScorerDefinition:
    return _fixed_policy_scorer(
        name="always_casual",
        description="Fixed policy baseline that always predicts casual exploratory use.",
        band=OversightLabel.CASUAL_EXPLORATORY,
    )


def always_assisted_scorer() -> ScorerDefinition:
    return _fixed_policy_scorer(
        name="always_assisted",
        description="Fixed policy baseline that always predicts assisted bounded use.",
        band=OversightLabel.ASSISTED_BOUNDED,
    )


def always_trained_review_scorer() -> ScorerDefinition:
    return _fixed_policy_scorer(
        name="always_trained_review",
        description="Fixed policy baseline that always predicts trained review.",
        band=OversightLabel.TRAINED_REVIEW_REQUIRED,
    )


def always_expert_review_scorer() -> ScorerDefinition:
    return _fixed_policy_scorer(
        name="always_expert_review",
        description="Fixed policy baseline that always predicts expert review.",
        band=OversightLabel.EXPERT_REVIEW_REQUIRED,
    )


def always_expert_led_scorer() -> ScorerDefinition:
    return _fixed_policy_scorer(
        name="always_expert_led",
        description="Fixed policy baseline that always predicts expert-led or no autonomous use.",
        band=OversightLabel.EXPERT_LED_OR_NO_AUTONOMOUS_USE,
    )


def reference_policy_scorers() -> tuple[ScorerDefinition, ...]:
    return (
        always_casual_scorer(),
        always_assisted_scorer(),
        always_trained_review_scorer(),
        always_expert_review_scorer(),
        always_expert_led_scorer(),
    )
```

- [ ] **Step 4: Export reference scorer API**

Update `core_model/__init__.py` imports:

```python
from .reference_scorers import (
    always_assisted_scorer,
    always_casual_scorer,
    always_expert_led_scorer,
    always_expert_review_scorer,
    always_trained_review_scorer,
    reference_policy_scorers,
)
```

Add to `__all__`:

```python
"always_assisted_scorer",
"always_casual_scorer",
"always_expert_led_scorer",
"always_expert_review_scorer",
"always_trained_review_scorer",
"reference_policy_scorers",
```

- [ ] **Step 5: Run tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_reference_scorers.py -v
```

Expected: all reference scorer tests pass.

- [ ] **Step 6: Run related tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_reference_scorers.py tests/test_model_comparison.py tests/test_evaluation_runner.py -v
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit core implementation**

```bash
git add core_model/reference_scorers.py core_model/__init__.py tests/test_reference_scorers.py
git commit -m "feat: add reference policy baselines"
```

---

### Task 2: Documentation and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/evidence-data-architecture.md`
- Modify: `docs/validation-agenda.md`

- [ ] **Step 1: Update README project structure**

In `README.md`, update the current implementation paragraph to include reference policy baselines:

```markdown
The active Python implementation lives in `core_model/`. It includes the numerical framework plus an evidence backbone for typed evidence records, source registry metadata, corpus management, local data loading, reproducible data snapshots, a provisional rubric scorer, deterministic reference policy baselines, an ordinal evaluation runner, a model-comparison layer, and calibration-oriented performance metrics.
```

Add to the `core_model/` tree:

```markdown
|   |-- reference_scorers.py       # Deterministic TEVV policy baselines
```

Add to the `tests/` tree:

```markdown
|   |-- test_reference_scorers.py  # Reference policy baseline tests
```

- [ ] **Step 2: Update evidence architecture**

In `docs/evidence-data-architecture.md`, after the paragraph describing `core_model/model_comparison.py`, add:

```markdown
Reference policy baselines in `core_model/reference_scorers.py` provide fixed-band TEVV controls such as always-casual, always-trained-review, and always-expert-led policies. These controls are not candidate risk models. They help determine whether a proposed model improves on trivial policies and make false-reassurance versus over-escalation tradeoffs visible before more complex Bayesian, Markov, statistical, or ensemble models are trusted.
```

In the implementation path list, add:

```markdown
- `core_model/reference_scorers.py` for deterministic TEVV policy baselines used in model comparison.
```

In the test list, add:

```markdown
- `tests/test_reference_scorers.py` for reference policy baseline behavior and model-comparison integration.
```

In the reference implementation paragraph, add `core_model/reference_scorers.py` alongside the other active backend modules.

- [ ] **Step 3: Update validation agenda**

In `docs/validation-agenda.md`, after the model comparison paragraph, add:

```markdown
Candidate models should be compared against trivial fixed policies before stronger performance claims are made. A useful model should either outperform reference policies on the relevant decision metrics or clearly explain a deliberate tradeoff, such as accepting more over-escalation to reduce false reassurance. Fixed policies should remain labeled as TEVV controls, not deployment recommendations.
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
git commit -m "docs: document reference policy baselines"
```

---

## Final Verification

- [ ] **Step 1: Run full tests**

```bash
.venv/bin/python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Manually compare baseline rubric against reference policies**

```bash
.venv/bin/python - <<'PY'
from core_model.evidence_io import load_corpus
from core_model.model_comparison import baseline_rubric_scorer, compare_models
from core_model.reference_scorers import reference_policy_scorers

corpus = load_corpus("data/examples/sources.json", "data/examples/evidence.jsonl")
report = compare_models(corpus, (baseline_rubric_scorer(), *reference_policy_scorers()))
for row in report.rows:
    print(row.scorer_name, row.common_metrics)
PY
```

Expected: output shows `baseline_rubric` plus all five reference policy rows with common metrics.

- [ ] **Step 3: Check exports**

```bash
.venv/bin/python - <<'PY'
import core_model
names = {
    "always_assisted_scorer",
    "always_casual_scorer",
    "always_expert_led_scorer",
    "always_expert_review_scorer",
    "always_trained_review_scorer",
    "reference_policy_scorers",
}
print({name: name in core_model.__all__ for name in sorted(names)})
PY
```

Expected: every printed value is `True`.

- [ ] **Step 4: Check status**

```bash
git status --short --branch
```

Expected: working tree is clean on the feature branch before merge.

## Success Checklist

- `core_model/reference_scorers.py` exists and returns five fixed-band policy baselines.
- Every reference policy is a `ScorerDefinition` with ordinal output type.
- Probability metrics are deferred for fixed policies.
- Reference policies run through `compare_models`.
- Tests expose expected false reassurance and over-escalation tradeoffs.
- Full test suite passes.
