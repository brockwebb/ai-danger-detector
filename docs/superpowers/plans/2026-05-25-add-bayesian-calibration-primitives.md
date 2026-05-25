# ADD Bayesian Calibration Primitives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add beta-binomial calibration primitives for bounded ADD probability parameters without creating a Bayesian scorer or calibration claim.

**Architecture:** Create `core_model/bayesian_calibration.py` with small frozen dataclasses for priors, observations, and posteriors. The module should be independent of scorer/model-comparison code and preserve source/evidence traceability for later TEVV review.

**Tech Stack:** Python 3.12-3.14, dataclasses, standard library math only, pytest.

---

## File Structure

- Create `core_model/bayesian_calibration.py`: beta prior, weighted beta-binomial observation, posterior update, summary.
- Modify `core_model/__init__.py`: export Bayesian calibration API.
- Create `tests/test_bayesian_calibration.py`: TDD coverage for validation, weighted updates, posterior formulas, traceability, and exports.
- Modify `README.md`: list the Bayesian calibration module and tests.
- Modify `docs/numerical-framework.md`: note that beta-binomial primitives now exist for bounded probabilities.
- Modify `docs/evidence-data-architecture.md`: document the primitive as calibration machinery, not a calibrated model.
- Modify `docs/validation-agenda.md`: name traceable Bayesian updates as part of later calibration work.

---

### Task 1: Beta-Binomial Calibration Primitives

**Files:**
- Create: `tests/test_bayesian_calibration.py`
- Create: `core_model/bayesian_calibration.py`
- Modify: `core_model/__init__.py`

- [ ] **Step 1: Write failing tests for priors, observations, updates, and exports**

Create `tests/test_bayesian_calibration.py`:

```python
import pytest

from core_model.bayesian_calibration import (
    BetaObservation,
    BetaPosterior,
    BetaPrior,
    update_beta_binomial,
)


def test_beta_prior_rejects_non_positive_parameters():
    with pytest.raises(ValueError, match="alpha must be positive"):
        BetaPrior(parameter_name="p_error_per_task", alpha=0, beta=2)

    with pytest.raises(ValueError, match="beta must be positive"):
        BetaPrior(parameter_name="p_error_per_task", alpha=2, beta=0)


def test_beta_observation_rejects_negative_counts_and_weights():
    with pytest.raises(ValueError, match="successes must be non-negative"):
        BetaObservation(successes=-1, failures=1)

    with pytest.raises(ValueError, match="failures must be non-negative"):
        BetaObservation(successes=1, failures=-1)

    with pytest.raises(ValueError, match="weight must be non-negative"):
        BetaObservation(successes=1, failures=1, weight=-0.5)


def test_update_beta_binomial_uses_unweighted_counts():
    prior = BetaPrior(
        parameter_name="p_error_per_task",
        alpha=2,
        beta=3,
        version="prior-v1",
    )
    observations = (
        BetaObservation(successes=3, failures=7, source_id="src-a"),
        BetaObservation(successes=1, failures=4, source_id="src-b"),
    )

    posterior = update_beta_binomial(prior, observations)

    assert isinstance(posterior, BetaPosterior)
    assert posterior.alpha == 6
    assert posterior.beta == 14
    assert posterior.effective_sample_size == 15
    assert posterior.mean == pytest.approx(0.3)
    assert posterior.variance == pytest.approx((6 * 14) / ((20**2) * 21))


def test_update_beta_binomial_applies_observation_weights():
    prior = BetaPrior(parameter_name="detectability", alpha=1, beta=1)
    observations = (
        BetaObservation(
            successes=8,
            failures=2,
            weight=0.25,
            source_id="src-weak",
            evidence_ids=("case-002", "case-001"),
            notes=("weak synthetic evidence",),
        ),
        BetaObservation(
            successes=4,
            failures=1,
            weight=2.0,
            source_id="src-strong",
            evidence_ids=("case-003",),
            notes=("adjudicated evidence",),
        ),
    )

    posterior = update_beta_binomial(prior, observations)

    assert posterior.alpha == pytest.approx(11.0)
    assert posterior.beta == pytest.approx(3.5)
    assert posterior.effective_sample_size == pytest.approx(12.5)
    assert posterior.source_ids == ("src-strong", "src-weak")
    assert posterior.evidence_ids == ("case-001", "case-002", "case-003")
    assert posterior.notes == ("weak synthetic evidence", "adjudicated evidence")


def test_posterior_summary_preserves_traceability_and_uncertainty():
    prior = BetaPrior(
        parameter_name="reversibility",
        alpha=4,
        beta=6,
        version="rev-prior-v1",
        notes=("expert elicited",),
    )
    observations = (
        BetaObservation(
            successes=2,
            failures=3,
            source_id="src-a",
            evidence_ids=("case-a",),
            notes=("retrospective review",),
        ),
    )

    posterior = update_beta_binomial(prior, observations)
    summary = posterior.summary()

    assert summary == {
        "parameter_name": "reversibility",
        "prior_version": "rev-prior-v1",
        "alpha": 6.0,
        "beta": 9.0,
        "mean": pytest.approx(0.4),
        "variance": pytest.approx((6 * 9) / ((15**2) * 16)),
        "effective_sample_size": 5.0,
        "source_ids": ("src-a",),
        "evidence_ids": ("case-a",),
        "notes": ("retrospective review",),
    }


def test_update_beta_binomial_rejects_empty_observations():
    prior = BetaPrior(parameter_name="p_error_per_task", alpha=1, beta=1)

    with pytest.raises(ValueError, match="at least one observation is required"):
        update_beta_binomial(prior, ())


def test_bayesian_calibration_exports_public_api():
    import core_model

    assert "BetaPrior" in core_model.__all__
    assert "BetaObservation" in core_model.__all__
    assert "BetaPosterior" in core_model.__all__
    assert "update_beta_binomial" in core_model.__all__
    assert core_model.BetaPrior is BetaPrior
    assert core_model.update_beta_binomial is update_beta_binomial
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_bayesian_calibration.py -v
```

Expected: FAIL during import because `core_model.bayesian_calibration` does not exist.

- [ ] **Step 3: Implement beta-binomial primitives**

Create `core_model/bayesian_calibration.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field


def _as_tuple(values: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    return tuple(str(value).strip() for value in values if str(value).strip())


@dataclass(frozen=True)
class BetaPrior:
    parameter_name: str
    alpha: float
    beta: float
    version: str = "unversioned"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not str(self.parameter_name).strip():
            raise ValueError("parameter_name must be non-empty")
        alpha = float(self.alpha)
        beta = float(self.beta)
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if beta <= 0:
            raise ValueError("beta must be positive")
        object.__setattr__(self, "parameter_name", str(self.parameter_name).strip())
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "version", str(self.version).strip() or "unversioned")
        object.__setattr__(self, "notes", _as_tuple(self.notes))


@dataclass(frozen=True)
class BetaObservation:
    successes: float
    failures: float
    weight: float = 1.0
    source_id: str | None = None
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        successes = float(self.successes)
        failures = float(self.failures)
        weight = float(self.weight)
        if successes < 0:
            raise ValueError("successes must be non-negative")
        if failures < 0:
            raise ValueError("failures must be non-negative")
        if weight < 0:
            raise ValueError("weight must be non-negative")
        source_id = None if self.source_id is None else str(self.source_id).strip()
        object.__setattr__(self, "successes", successes)
        object.__setattr__(self, "failures", failures)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "source_id", source_id or None)
        object.__setattr__(self, "evidence_ids", _as_tuple(self.evidence_ids))
        object.__setattr__(self, "notes", _as_tuple(self.notes))

    @property
    def weighted_successes(self) -> float:
        return self.successes * self.weight

    @property
    def weighted_failures(self) -> float:
        return self.failures * self.weight

    @property
    def effective_sample_size(self) -> float:
        return (self.successes + self.failures) * self.weight


@dataclass(frozen=True)
class BetaPosterior:
    alpha: float
    beta: float
    prior: BetaPrior
    observations: tuple[BetaObservation, ...]
    effective_sample_size: float
    source_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / ((total**2) * (total + 1.0))

    def summary(self) -> dict:
        return {
            "parameter_name": self.prior.parameter_name,
            "prior_version": self.prior.version,
            "alpha": self.alpha,
            "beta": self.beta,
            "mean": self.mean,
            "variance": self.variance,
            "effective_sample_size": self.effective_sample_size,
            "source_ids": self.source_ids,
            "evidence_ids": self.evidence_ids,
            "notes": self.notes,
        }


def update_beta_binomial(
    prior: BetaPrior,
    observations: tuple[BetaObservation, ...] | list[BetaObservation],
) -> BetaPosterior:
    observations = tuple(observations)
    if not observations:
        raise ValueError("at least one observation is required")
    alpha = prior.alpha + sum(observation.weighted_successes for observation in observations)
    beta = prior.beta + sum(observation.weighted_failures for observation in observations)
    effective_sample_size = sum(
        observation.effective_sample_size for observation in observations
    )
    source_ids = tuple(
        sorted(
            {
                observation.source_id
                for observation in observations
                if observation.source_id is not None
            }
        )
    )
    evidence_ids = tuple(
        sorted(
            {
                evidence_id
                for observation in observations
                for evidence_id in observation.evidence_ids
            }
        )
    )
    notes = tuple(note for observation in observations for note in observation.notes)
    return BetaPosterior(
        alpha=alpha,
        beta=beta,
        prior=prior,
        observations=observations,
        effective_sample_size=effective_sample_size,
        source_ids=source_ids,
        evidence_ids=evidence_ids,
        notes=notes,
    )
```

- [ ] **Step 4: Export Bayesian calibration API**

Update `core_model/__init__.py` imports:

```python
from .bayesian_calibration import (
    BetaObservation,
    BetaPosterior,
    BetaPrior,
    update_beta_binomial,
)
```

Add to `__all__`:

```python
"BetaObservation",
"BetaPosterior",
"BetaPrior",
"update_beta_binomial",
```

- [ ] **Step 5: Run tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_bayesian_calibration.py -v
```

Expected: all Bayesian calibration tests pass.

- [ ] **Step 6: Run related tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_bayesian_calibration.py tests/test_performance_metrics.py tests/test_numerical_framework.py -v
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit core implementation**

```bash
git add core_model/bayesian_calibration.py core_model/__init__.py tests/test_bayesian_calibration.py
git commit -m "feat: add beta-binomial calibration primitives"
```

---

### Task 2: Documentation and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/numerical-framework.md`
- Modify: `docs/evidence-data-architecture.md`
- Modify: `docs/validation-agenda.md`

- [ ] **Step 1: Update README project structure**

In `README.md`, update the current implementation paragraph to include Bayesian calibration primitives:

```markdown
The active Python implementation lives in `core_model/`. It includes the numerical framework plus an evidence backbone for typed evidence records, source registry metadata, corpus management, local data loading, reproducible data snapshots, beta-binomial calibration primitives, a provisional rubric scorer, deterministic reference policy baselines, an ordinal evaluation runner, a model-comparison layer, and calibration-oriented performance metrics.
```

Add to the `core_model/` tree:

```markdown
|   |-- bayesian_calibration.py    # Beta-binomial calibration primitives
```

Add to the `tests/` tree:

```markdown
|   |-- test_bayesian_calibration.py # Bayesian calibration primitive tests
```

- [ ] **Step 2: Update numerical framework**

In `docs/numerical-framework.md`, after the paragraph ending with "Calibration should preserve disagreement when experts do not agree. A wide or multimodal prior can be more honest than forcing a single consensus value.", add:

```markdown
The first Bayesian calibration implementation lives in `core_model/bayesian_calibration.py`. It provides beta-binomial update primitives for bounded probabilities with traceable priors, weighted observations, source IDs, evidence IDs, and posterior summaries. It is calibration machinery only: it does not create a Bayesian scorer, does not perform MCMC, and does not model unbounded event rates such as `lambda_error_rate`.
```

- [ ] **Step 3: Update evidence data architecture**

In `docs/evidence-data-architecture.md`, after the reference policy baseline paragraph, add:

```markdown
The beta-binomial primitives in `core_model/bayesian_calibration.py` are the first Bayesian calibration machinery. They update bounded probability parameters from traceable observations and preserve source IDs, evidence IDs, weights, and notes. They are not a calibrated ADD model by themselves and should not be used for unbounded rates or exposure-based event counts.
```

In the implementation path list, add:

```markdown
- `core_model/bayesian_calibration.py` for beta-binomial priors, weighted observations, posterior updates, and traceable summaries.
```

In the test list, add:

```markdown
- `tests/test_bayesian_calibration.py` for beta-binomial validation, weighted updates, posterior summaries, and traceability.
```

In the reference implementation paragraph, add `core_model/bayesian_calibration.py` alongside the other active backend modules.

- [ ] **Step 4: Update validation agenda**

In `docs/validation-agenda.md`, after the model comparison/fixed policy paragraphs, add:

```markdown
Bayesian updates should preserve traceability from posterior summaries back to priors, source IDs, evidence IDs, observation weights, and notes. Early beta-binomial updates should be treated as calibration machinery for bounded probabilities, not as proof that ADD is calibrated. Rate or exposure models should use separate count-model assumptions rather than forcing unbounded rates into beta priors.
```

- [ ] **Step 5: Run full verification and scans**

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

- [ ] **Step 6: Commit documentation updates**

```bash
git add README.md docs/numerical-framework.md docs/evidence-data-architecture.md docs/validation-agenda.md
git commit -m "docs: document bayesian calibration primitives"
```

---

## Final Verification

- [ ] **Step 1: Run full tests**

```bash
.venv/bin/python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Manually run a beta-binomial update**

```bash
.venv/bin/python - <<'PY'
from core_model.bayesian_calibration import BetaObservation, BetaPrior, update_beta_binomial

prior = BetaPrior(parameter_name="p_error_per_task", alpha=2, beta=8, version="demo-v1")
posterior = update_beta_binomial(
    prior,
    (
        BetaObservation(successes=3, failures=7, source_id="src-a", evidence_ids=("case-a",)),
        BetaObservation(successes=2, failures=8, weight=0.5, source_id="src-b", evidence_ids=("case-b",)),
    ),
)
print(posterior.summary())
PY
```

Expected: output shows posterior alpha/beta, mean, variance, effective sample size, source IDs, and evidence IDs.

- [ ] **Step 3: Check exports**

```bash
.venv/bin/python - <<'PY'
import core_model
names = {
    "BetaObservation",
    "BetaPosterior",
    "BetaPrior",
    "update_beta_binomial",
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

- `core_model/bayesian_calibration.py` exists and implements beta-binomial prior, observation, posterior, and update primitives.
- Invalid priors, counts, and weights are rejected.
- Weighted observations update alpha, beta, and effective sample size correctly.
- Posterior summaries preserve source IDs, evidence IDs, notes, mean, and variance.
- Docs state that this is calibration machinery, not a calibrated model or MCMC implementation.
- Full test suite passes.
