# ADD Numerical Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first tested reference implementation of `docs/numerical-framework.md` and archive older exploratory prototype material.

**Architecture:** Keep the active implementation small and auditable in `core_model/numerical_framework.py`. Archive older prototype modules under `archive/exploratory-prototype/` so the public repository has a clear distinction between historical exploration and the current reference model. Use tests to define the numerical API before implementation.

**Tech Stack:** Python 3.12-3.14, NumPy for sampling and matrix calculations, pytest for tests, Markdown docs.

---

## File Structure

- Create `archive/exploratory-prototype/README.md`: historical status note for archived code.
- Move `applications/`, `data_generation/`, `machine_learning/`, `model_validation/`, `visualization/`, old `core_model/*.py`, and `parameter_sensitivity.png` into `archive/exploratory-prototype/` while preserving their relative layout.
- Keep `core_model/` active with new `__init__.py` and `numerical_framework.py`.
- Create `tests/test_numerical_framework.py`: behavior tests for the new active API.
- Create `pyproject.toml`: package metadata, Python range, runtime and test dependencies.
- Modify `README.md`: update repository structure and archived prototype language.
- Modify `docs/numerical-framework.md`: link to implementation and clarify Monte Carlo over Markov workflow versus strict MCMC.

---

### Task 1: Archive Exploratory Prototype

**Files:**
- Create: `archive/exploratory-prototype/README.md`
- Move: `applications/` to `archive/exploratory-prototype/applications/`
- Move: `data_generation/` to `archive/exploratory-prototype/data_generation/`
- Move: `machine_learning/` to `archive/exploratory-prototype/machine_learning/`
- Move: `model_validation/` to `archive/exploratory-prototype/model_validation/`
- Move: `visualization/` to `archive/exploratory-prototype/visualization/`
- Move: `core_model/model_definition.py` to `archive/exploratory-prototype/core_model/model_definition.py`
- Move: `core_model/domain_profiles.py` to `archive/exploratory-prototype/core_model/domain_profiles.py`
- Move: `core_model/multi_regime_model.py` to `archive/exploratory-prototype/core_model/multi_regime_model.py`
- Move: `parameter_sensitivity.png` to `archive/exploratory-prototype/parameter_sensitivity.png`
- Modify later: `core_model/__init__.py`

- [ ] **Step 1: Create the archive directory and status README**

Use `mkdir -p archive/exploratory-prototype/core_model`.

Create `archive/exploratory-prototype/README.md` with:

```markdown
# Exploratory Prototype Archive

This directory preserves earlier AI Danger Detector prototype work for historical reference.

The archived files are exploratory and are not the active reference implementation. They may contain provisional formulas, old assumptions, ad hoc scripts, notebook-era dependencies, and unvalidated parameter choices.

For current work, start with:

- `docs/whitepaper.md`
- `docs/model-rubric.md`
- `docs/numerical-framework.md`
- `docs/validation-agenda.md`
- `core_model/numerical_framework.py`

The archive is useful for seeing project history, sensitivity experiments, and earlier modeling attempts. It should not be used as an operational safety, legal, medical, financial, or compliance tool.
```

- [ ] **Step 2: Move historical directories and files**

Run:

```bash
git mv applications archive/exploratory-prototype/applications
git mv data_generation archive/exploratory-prototype/data_generation
git mv machine_learning archive/exploratory-prototype/machine_learning
git mv model_validation archive/exploratory-prototype/model_validation
git mv visualization archive/exploratory-prototype/visualization
git mv core_model/model_definition.py archive/exploratory-prototype/core_model/model_definition.py
git mv core_model/domain_profiles.py archive/exploratory-prototype/core_model/domain_profiles.py
git mv core_model/multi_regime_model.py archive/exploratory-prototype/core_model/multi_regime_model.py
git mv parameter_sensitivity.png archive/exploratory-prototype/parameter_sensitivity.png
```

- [ ] **Step 3: Verify active tree is cleanly separated**

Run:

```bash
find core_model -maxdepth 1 -type f -print | sort
find archive/exploratory-prototype -maxdepth 2 -type d -print | sort
```

Expected:

- `core_model/__init__.py` remains active.
- Archived historical directories appear under `archive/exploratory-prototype/`.

- [ ] **Step 4: Commit archive move**

```bash
git add archive core_model
git commit -m "chore: archive exploratory prototype"
```

---

### Task 2: Package Metadata

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Create package metadata**

Create `pyproject.toml` with:

```toml
[project]
name = "ai-danger-detector"
version = "0.1.0"
description = "Prototype framework for AI oversight triage and numerical reliance workflow modeling."
readme = "README.md"
requires-python = ">=3.12,<3.15"
license = "MIT"
dependencies = [
    "numpy>=2.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

- [ ] **Step 2: Verify package metadata**

Run:

```bash
.venv/bin/python -m pytest --version
.venv/bin/python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb')); print('pyproject ok')"
```

Expected:

- pytest version prints.
- `pyproject ok` prints.

- [ ] **Step 3: Commit package metadata**

```bash
git add pyproject.toml
git commit -m "chore: add Python project metadata"
```

---

### Task 3: Transition Matrix Validation

**Files:**
- Create: `tests/test_numerical_framework.py`
- Create: `core_model/numerical_framework.py`
- Modify: `core_model/__init__.py`

- [ ] **Step 1: Write failing tests for state exports and transition validation**

Create `tests/test_numerical_framework.py` with:

```python
import pytest

from core_model.numerical_framework import (
    MarkovState,
    validate_transition_matrix,
)


def _valid_matrix():
    return {
        MarkovState.S0: {MarkovState.S1: 1.0},
        MarkovState.S1: {MarkovState.S2: 0.2, MarkovState.S3: 0.4, MarkovState.S4: 0.4},
        MarkovState.S2: {MarkovState.S6: 1.0},
        MarkovState.S3: {MarkovState.S5: 0.5, MarkovState.S4: 0.25, MarkovState.S6: 0.25},
        MarkovState.S4: {MarkovState.S5: 0.7, MarkovState.S6: 0.3},
        MarkovState.S5: {MarkovState.S7: 0.8, MarkovState.S1: 0.2},
        MarkovState.S6: {MarkovState.S7: 0.9, MarkovState.S8: 0.1},
        MarkovState.S7: {MarkovState.S7: 1.0},
        MarkovState.S8: {MarkovState.S8: 1.0},
    }


def test_all_markov_states_are_named_s0_through_s8():
    assert [state.value for state in MarkovState] == [f"S{i}" for i in range(9)]


def test_validate_transition_matrix_accepts_complete_matrix():
    normalized = validate_transition_matrix(_valid_matrix())

    assert normalized[MarkovState.S0][MarkovState.S1] == pytest.approx(1.0)


def test_validate_transition_matrix_rejects_missing_required_state():
    matrix = _valid_matrix()
    del matrix[MarkovState.S3]

    with pytest.raises(ValueError, match="missing rows"):
        validate_transition_matrix(matrix)


def test_validate_transition_matrix_rejects_negative_probability():
    matrix = _valid_matrix()
    matrix[MarkovState.S1] = {MarkovState.S2: -0.1, MarkovState.S3: 0.6, MarkovState.S4: 0.5}

    with pytest.raises(ValueError, match="negative"):
        validate_transition_matrix(matrix)


def test_validate_transition_matrix_rejects_row_that_does_not_sum_to_one():
    matrix = _valid_matrix()
    matrix[MarkovState.S1] = {MarkovState.S2: 0.2, MarkovState.S3: 0.2, MarkovState.S4: 0.2}

    with pytest.raises(ValueError, match="sum to 1"):
        validate_transition_matrix(matrix)
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_numerical_framework.py -v
```

Expected: FAIL because `core_model.numerical_framework` does not exist.

- [ ] **Step 3: Implement minimal state enum and matrix validation**

Create `core_model/numerical_framework.py` with:

```python
from __future__ import annotations

from enum import Enum
from typing import Mapping


class MarkovState(str, Enum):
    S0 = "S0"
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"
    S5 = "S5"
    S6 = "S6"
    S7 = "S7"
    S8 = "S8"


TransitionMatrix = dict[MarkovState, dict[MarkovState, float]]
TERMINAL_STATES = (MarkovState.S7, MarkovState.S8)


def _coerce_state(value: MarkovState | str) -> MarkovState:
    try:
        return value if isinstance(value, MarkovState) else MarkovState(value)
    except ValueError as exc:
        raise ValueError(f"unknown Markov state: {value!r}") from exc


def validate_transition_matrix(
    matrix: Mapping[MarkovState | str, Mapping[MarkovState | str, float]],
    *,
    tolerance: float = 1e-9,
) -> TransitionMatrix:
    normalized: TransitionMatrix = {}

    for raw_from_state, raw_edges in matrix.items():
        from_state = _coerce_state(raw_from_state)
        if not raw_edges:
            raise ValueError(f"transition row for {from_state.value} is empty")

        row: dict[MarkovState, float] = {}
        for raw_to_state, probability in raw_edges.items():
            to_state = _coerce_state(raw_to_state)
            probability = float(probability)
            if probability < 0:
                raise ValueError(f"transition row for {from_state.value} contains a negative probability")
            row[to_state] = probability

        row_total = sum(row.values())
        if abs(row_total - 1.0) > tolerance:
            raise ValueError(f"transition row for {from_state.value} must sum to 1")
        normalized[from_state] = row

    missing_rows = [state.value for state in MarkovState if state not in normalized]
    if missing_rows:
        raise ValueError(f"transition matrix missing rows: {', '.join(missing_rows)}")

    for terminal_state in TERMINAL_STATES:
        if normalized[terminal_state] != {terminal_state: 1.0}:
            raise ValueError(f"terminal state {terminal_state.value} must self-loop with probability 1")

    return normalized
```

Replace `core_model/__init__.py` with:

```python
"""Active AI Danger Detector numerical framework API."""

from .numerical_framework import MarkovState, validate_transition_matrix

__all__ = ["MarkovState", "validate_transition_matrix"]
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_numerical_framework.py -v
```

Expected: all current tests pass.

- [ ] **Step 5: Commit transition validation**

```bash
git add core_model tests
git commit -m "feat: add Markov transition validation"
```

---

### Task 4: Markov Workflow Evaluation

**Files:**
- Modify: `tests/test_numerical_framework.py`
- Modify: `core_model/numerical_framework.py`
- Modify: `core_model/__init__.py`

- [ ] **Step 1: Add failing tests for workflow probabilities**

Append to `tests/test_numerical_framework.py`:

```python
from core_model.numerical_framework import evaluate_markov_workflow


def test_evaluate_markov_workflow_returns_terminal_probabilities():
    result = evaluate_markov_workflow(_valid_matrix())

    assert result.terminal_probabilities[MarkovState.S7] == pytest.approx(0.8911764706)
    assert result.terminal_probabilities[MarkovState.S8] == pytest.approx(0.1088235294)
    assert sum(result.terminal_probabilities.values()) == pytest.approx(1.0)


def test_evaluate_markov_workflow_tracks_probability_of_unverified_action_path():
    result = evaluate_markov_workflow(_valid_matrix())

    assert result.unverified_action_probability == pytest.approx(0.2)
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_numerical_framework.py -v
```

Expected: FAIL because `evaluate_markov_workflow` is not defined.

- [ ] **Step 3: Implement analytical absorbing-chain evaluation**

Add imports and dataclass:

```python
from dataclasses import dataclass

import numpy as np
```

Add:

```python
@dataclass(frozen=True)
class WorkflowResult:
    terminal_probabilities: dict[MarkovState, float]
    unverified_action_probability: float
```

Add:

```python
def _hitting_probability(
    matrix: TransitionMatrix,
    *,
    start_state: MarkovState,
    hit_state: MarkovState,
) -> float:
    absorbing = set(TERMINAL_STATES) | {hit_state}
    transient_states = [state for state in MarkovState if state not in absorbing]
    index = {state: position for position, state in enumerate(transient_states)}

    if start_state == hit_state:
        return 1.0
    if start_state in TERMINAL_STATES:
        return 0.0

    q = np.zeros((len(transient_states), len(transient_states)))
    r = np.zeros(len(transient_states))

    for from_state in transient_states:
        row = index[from_state]
        for to_state, probability in matrix[from_state].items():
            if to_state == hit_state:
                r[row] += probability
            elif to_state in index:
                q[row, index[to_state]] += probability

    solution = np.linalg.solve(np.eye(len(transient_states)) - q, r)
    return float(solution[index[start_state]])


def evaluate_markov_workflow(
    matrix: Mapping[MarkovState | str, Mapping[MarkovState | str, float]],
    *,
    start_state: MarkovState | str = MarkovState.S0,
) -> WorkflowResult:
    normalized = validate_transition_matrix(matrix)
    start = _coerce_state(start_state)
    transient_states = [state for state in MarkovState if state not in TERMINAL_STATES]
    terminal_states = list(TERMINAL_STATES)
    transient_index = {state: position for position, state in enumerate(transient_states)}
    terminal_index = {state: position for position, state in enumerate(terminal_states)}

    q = np.zeros((len(transient_states), len(transient_states)))
    r = np.zeros((len(transient_states), len(terminal_states)))

    for from_state in transient_states:
        row = transient_index[from_state]
        for to_state, probability in normalized[from_state].items():
            if to_state in terminal_index:
                r[row, terminal_index[to_state]] += probability
            else:
                q[row, transient_index[to_state]] += probability

    absorption = np.linalg.solve(np.eye(len(transient_states)) - q, r)
    if start in terminal_index:
        terminal_probabilities = {state: 0.0 for state in terminal_states}
        terminal_probabilities[start] = 1.0
    else:
        terminal_probabilities = {
            terminal_state: float(absorption[transient_index[start], terminal_index[terminal_state]])
            for terminal_state in terminal_states
        }

    return WorkflowResult(
        terminal_probabilities=terminal_probabilities,
        unverified_action_probability=_hitting_probability(
            normalized,
            start_state=start,
            hit_state=MarkovState.S2,
        ),
    )
```

Update `core_model/__init__.py` to export `WorkflowResult` and `evaluate_markov_workflow`.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_numerical_framework.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit Markov workflow evaluation**

```bash
git add core_model tests
git commit -m "feat: evaluate Markov workflow outcomes"
```

---

### Task 5: Distribution Specs and Scenario Construction

**Files:**
- Modify: `tests/test_numerical_framework.py`
- Modify: `core_model/numerical_framework.py`
- Modify: `core_model/__init__.py`

- [ ] **Step 1: Add failing tests for distributions and scenario transition generation**

Append to `tests/test_numerical_framework.py`:

```python
from core_model.numerical_framework import (
    DistributionSpec,
    ScenarioSpec,
    build_transition_matrix,
)


def test_distribution_spec_samples_fixed_beta_triangular_and_lognormal_values():
    rng = __import__("numpy").random.default_rng(123)

    fixed = DistributionSpec.fixed(0.4).sample(rng, 4)
    beta = DistributionSpec.beta(alpha=2, beta=5).sample(rng, 200)
    triangular = DistributionSpec.triangular(left=0.1, mode=0.3, right=0.9).sample(rng, 200)
    lognormal = DistributionSpec.lognormal(mean=1.0, sigma=0.25).sample(rng, 200)

    assert fixed.tolist() == [0.4, 0.4, 0.4, 0.4]
    assert beta.min() >= 0.0
    assert beta.max() <= 1.0
    assert triangular.min() >= 0.1
    assert triangular.max() <= 0.9
    assert lognormal.min() > 0.0


def test_build_transition_matrix_uses_scenario_values_to_create_valid_matrix():
    scenario = ScenarioSpec(
        error_probability=DistributionSpec.fixed(0.2),
        severity=DistributionSpec.fixed(0.7),
        detectability=DistributionSpec.fixed(0.6),
        reversibility=DistributionSpec.fixed(0.5),
        verification_burden=DistributionSpec.fixed(0.8),
        governance_strength=DistributionSpec.fixed(0.7),
        conditional_loss=DistributionSpec.fixed(1000.0),
    )

    sampled = scenario.sample(__import__("numpy").random.default_rng(123))
    matrix = build_transition_matrix(sampled)
    normalized = validate_transition_matrix(matrix)

    assert normalized[MarkovState.S0][MarkovState.S1] == pytest.approx(1.0)
    assert normalized[MarkovState.S6][MarkovState.S8] > 0.0
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_numerical_framework.py -v
```

Expected: FAIL because `DistributionSpec`, `ScenarioSpec`, and `build_transition_matrix` are not defined.

- [ ] **Step 3: Implement distribution specs and scenario sampling**

Add:

```python
from typing import Literal
```

Add:

```python
@dataclass(frozen=True)
class DistributionSpec:
    kind: Literal["fixed", "beta", "triangular", "lognormal"]
    parameters: dict[str, float]

    @classmethod
    def fixed(cls, value: float) -> "DistributionSpec":
        return cls("fixed", {"value": float(value)})

    @classmethod
    def beta(cls, *, alpha: float, beta: float) -> "DistributionSpec":
        return cls("beta", {"alpha": float(alpha), "beta": float(beta)})

    @classmethod
    def triangular(cls, *, left: float, mode: float, right: float) -> "DistributionSpec":
        return cls("triangular", {"left": float(left), "mode": float(mode), "right": float(right)})

    @classmethod
    def lognormal(cls, *, mean: float, sigma: float) -> "DistributionSpec":
        return cls("lognormal", {"mean": float(mean), "sigma": float(sigma)})

    def sample(self, rng: np.random.Generator, size: int | None = None) -> np.ndarray:
        if self.kind == "fixed":
            shape = () if size is None else size
            return np.full(shape, self.parameters["value"], dtype=float)
        if self.kind == "beta":
            return rng.beta(self.parameters["alpha"], self.parameters["beta"], size=size)
        if self.kind == "triangular":
            return rng.triangular(
                self.parameters["left"],
                self.parameters["mode"],
                self.parameters["right"],
                size=size,
            )
        if self.kind == "lognormal":
            return rng.lognormal(self.parameters["mean"], self.parameters["sigma"], size=size)
        raise ValueError(f"unknown distribution kind: {self.kind}")
```

Add:

```python
@dataclass(frozen=True)
class SampledScenario:
    error_probability: float
    severity: float
    detectability: float
    reversibility: float
    verification_burden: float
    governance_strength: float
    conditional_loss: float


@dataclass(frozen=True)
class ScenarioSpec:
    error_probability: DistributionSpec
    severity: DistributionSpec
    detectability: DistributionSpec
    reversibility: DistributionSpec
    verification_burden: DistributionSpec
    governance_strength: DistributionSpec
    conditional_loss: DistributionSpec

    def sample(self, rng: np.random.Generator) -> SampledScenario:
        return SampledScenario(
            error_probability=float(self.error_probability.sample(rng)),
            severity=float(self.severity.sample(rng)),
            detectability=float(self.detectability.sample(rng)),
            reversibility=float(self.reversibility.sample(rng)),
            verification_burden=float(self.verification_burden.sample(rng)),
            governance_strength=float(self.governance_strength.sample(rng)),
            conditional_loss=float(self.conditional_loss.sample(rng)),
        )
```

Add helpers:

```python
def _clamp_probability(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _normalize_weights(weights: Mapping[MarkovState, float]) -> dict[MarkovState, float]:
    cleaned = {state: max(float(weight), 0.0) for state, weight in weights.items()}
    total = sum(cleaned.values())
    if total <= 0:
        share = 1.0 / len(cleaned)
        return {state: share for state in cleaned}
    return {state: weight / total for state, weight in cleaned.items()}
```

Add:

```python
def build_transition_matrix(sampled: SampledScenario) -> TransitionMatrix:
    error_probability = _clamp_probability(sampled.error_probability)
    severity = _clamp_probability(sampled.severity)
    detectability = _clamp_probability(sampled.detectability)
    reversibility = _clamp_probability(sampled.reversibility)
    verification_burden = _clamp_probability(sampled.verification_burden)
    governance_strength = _clamp_probability(sampled.governance_strength)

    first_review = _normalize_weights(
        {
            MarkovState.S2: (1.0 - governance_strength) * (1.0 - 0.4 * detectability),
            MarkovState.S3: (0.25 + detectability) * (1.0 - 0.5 * governance_strength),
            MarkovState.S4: (0.15 + governance_strength) * (0.5 + 0.5 * verification_burden),
        }
    )

    checking_detection = _clamp_probability(0.15 + 0.65 * detectability - 0.25 * verification_burden)
    checking_escalation = _clamp_probability(0.10 + 0.45 * governance_strength + 0.20 * verification_burden)
    checking_action = max(0.0, 1.0 - checking_detection - checking_escalation)
    s3 = _normalize_weights(
        {
            MarkovState.S5: checking_detection,
            MarkovState.S4: checking_escalation,
            MarkovState.S6: checking_action,
        }
    )

    expert_detection = _clamp_probability(0.45 + 0.45 * detectability + 0.10 * governance_strength)
    s4 = {MarkovState.S5: expert_detection, MarkovState.S6: 1.0 - expert_detection}

    harm_after_action = _clamp_probability(
        error_probability * (0.25 + 0.75 * severity) * (1.0 - 0.55 * reversibility)
    )

    return validate_transition_matrix(
        {
            MarkovState.S0: {MarkovState.S1: 1.0},
            MarkovState.S1: first_review,
            MarkovState.S2: {MarkovState.S6: 1.0},
            MarkovState.S3: s3,
            MarkovState.S4: s4,
            MarkovState.S5: {MarkovState.S7: 0.85, MarkovState.S1: 0.15},
            MarkovState.S6: {MarkovState.S7: 1.0 - harm_after_action, MarkovState.S8: harm_after_action},
            MarkovState.S7: {MarkovState.S7: 1.0},
            MarkovState.S8: {MarkovState.S8: 1.0},
        }
    )
```

Update `core_model/__init__.py` to export `DistributionSpec`, `ScenarioSpec`, and `build_transition_matrix`.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_numerical_framework.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit distribution and scenario support**

```bash
git add core_model tests
git commit -m "feat: add scenario sampling for numerical framework"
```

---

### Task 6: Monte Carlo Simulation and Summaries

**Files:**
- Modify: `tests/test_numerical_framework.py`
- Modify: `core_model/numerical_framework.py`
- Modify: `core_model/__init__.py`

- [ ] **Step 1: Add failing tests for simulation reproducibility, risk ordering, and summary shape**

Append to `tests/test_numerical_framework.py`:

```python
from core_model.numerical_framework import (
    SimulationConfig,
    run_monte_carlo,
    summarize_simulation,
)


def _scenario(error, severity, detectability, reversibility, burden, governance, loss):
    return ScenarioSpec(
        error_probability=DistributionSpec.fixed(error),
        severity=DistributionSpec.fixed(severity),
        detectability=DistributionSpec.fixed(detectability),
        reversibility=DistributionSpec.fixed(reversibility),
        verification_burden=DistributionSpec.fixed(burden),
        governance_strength=DistributionSpec.fixed(governance),
        conditional_loss=DistributionSpec.fixed(loss),
    )


def test_run_monte_carlo_is_reproducible_with_seed():
    scenario = _scenario(0.2, 0.7, 0.6, 0.5, 0.8, 0.7, 1000.0)
    config = SimulationConfig(iterations=50, seed=123)

    first = run_monte_carlo(scenario, config)
    second = run_monte_carlo(scenario, config)

    assert first.oversight_scores.tolist() == second.oversight_scores.tolist()
    assert first.realized_harm_probabilities.tolist() == second.realized_harm_probabilities.tolist()


def test_higher_risk_scenario_increases_expert_led_probability():
    low = run_monte_carlo(
        _scenario(0.03, 0.2, 0.9, 0.9, 0.2, 0.8, 100.0),
        SimulationConfig(iterations=50, seed=123),
    )
    high = run_monte_carlo(
        _scenario(0.5, 0.95, 0.15, 0.1, 0.95, 0.2, 10000.0),
        SimulationConfig(iterations=50, seed=123),
    )

    assert high.threshold_probabilities["expert_led_or_no_autonomous_use"] > low.threshold_probabilities["expert_led_or_no_autonomous_use"]
    assert high.expected_losses.mean() > low.expected_losses.mean()


def test_summarize_simulation_reports_documented_outputs():
    result = run_monte_carlo(
        _scenario(0.2, 0.7, 0.6, 0.5, 0.8, 0.7, 1000.0),
        SimulationConfig(iterations=50, seed=123),
    )

    summary = summarize_simulation(result)

    assert set(summary) == {
        "median_oversight_score",
        "oversight_score_interval",
        "p_trained_review_required",
        "p_expert_review_required",
        "p_expert_led_or_no_autonomous_use",
        "p_unverified_action",
        "p_realized_harm",
        "expected_loss",
    }
    assert summary["expected_loss"] > 0.0
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_numerical_framework.py -v
```

Expected: FAIL because simulation classes and functions are not defined.

- [ ] **Step 3: Implement simulation dataclasses and scoring**

Add:

```python
@dataclass(frozen=True)
class SimulationConfig:
    iterations: int = 1000
    seed: int | None = None
    trained_review_threshold: float = 0.35
    expert_review_threshold: float = 0.60
    expert_led_threshold: float = 0.80
    interval: tuple[float, float] = (0.05, 0.95)
```

Add:

```python
@dataclass(frozen=True)
class SimulationResult:
    oversight_scores: np.ndarray
    realized_harm_probabilities: np.ndarray
    unverified_action_probabilities: np.ndarray
    expected_losses: np.ndarray
    threshold_probabilities: dict[str, float]
```

Add:

```python
def calculate_oversight_score(sampled: SampledScenario) -> float:
    return _clamp_probability(
        0.22 * _clamp_probability(sampled.error_probability)
        + 0.25 * _clamp_probability(sampled.severity)
        + 0.18 * (1.0 - _clamp_probability(sampled.detectability))
        + 0.15 * (1.0 - _clamp_probability(sampled.reversibility))
        + 0.15 * _clamp_probability(sampled.verification_burden)
        + 0.05 * (1.0 - _clamp_probability(sampled.governance_strength))
    )
```

Add:

```python
def run_monte_carlo(scenario: ScenarioSpec, config: SimulationConfig | None = None) -> SimulationResult:
    config = config or SimulationConfig()
    if config.iterations <= 0:
        raise ValueError("iterations must be positive")

    rng = np.random.default_rng(config.seed)
    oversight_scores = np.zeros(config.iterations)
    realized_harm_probabilities = np.zeros(config.iterations)
    unverified_action_probabilities = np.zeros(config.iterations)
    expected_losses = np.zeros(config.iterations)

    for index in range(config.iterations):
        sampled = scenario.sample(rng)
        oversight_score = calculate_oversight_score(sampled)
        workflow = evaluate_markov_workflow(build_transition_matrix(sampled))
        realized_harm = workflow.terminal_probabilities[MarkovState.S8]

        oversight_scores[index] = oversight_score
        realized_harm_probabilities[index] = realized_harm
        unverified_action_probabilities[index] = workflow.unverified_action_probability
        expected_losses[index] = realized_harm * max(sampled.conditional_loss, 0.0)

    return SimulationResult(
        oversight_scores=oversight_scores,
        realized_harm_probabilities=realized_harm_probabilities,
        unverified_action_probabilities=unverified_action_probabilities,
        expected_losses=expected_losses,
        threshold_probabilities={
            "trained_review_required": float(np.mean(oversight_scores >= config.trained_review_threshold)),
            "expert_review_required": float(np.mean(oversight_scores >= config.expert_review_threshold)),
            "expert_led_or_no_autonomous_use": float(np.mean(oversight_scores >= config.expert_led_threshold)),
        },
    )
```

Add:

```python
def summarize_simulation(result: SimulationResult, *, interval: tuple[float, float] = (0.05, 0.95)) -> dict[str, float | tuple[float, float]]:
    lower, upper = np.quantile(result.oversight_scores, interval)
    return {
        "median_oversight_score": float(np.median(result.oversight_scores)),
        "oversight_score_interval": (float(lower), float(upper)),
        "p_trained_review_required": result.threshold_probabilities["trained_review_required"],
        "p_expert_review_required": result.threshold_probabilities["expert_review_required"],
        "p_expert_led_or_no_autonomous_use": result.threshold_probabilities["expert_led_or_no_autonomous_use"],
        "p_unverified_action": float(np.mean(result.unverified_action_probabilities)),
        "p_realized_harm": float(np.mean(result.realized_harm_probabilities)),
        "expected_loss": float(np.mean(result.expected_losses)),
    }
```

Update `core_model/__init__.py` to export `SimulationConfig`, `SimulationResult`, `calculate_oversight_score`, `run_monte_carlo`, and `summarize_simulation`.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_numerical_framework.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit Monte Carlo simulation**

```bash
git add core_model tests
git commit -m "feat: add Monte Carlo oversight simulation"
```

---

### Task 7: Documentation and Full Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/numerical-framework.md`

- [ ] **Step 1: Update README project structure**

Update README so the project structure shows:

```text
AI-Danger-Detector/
|
|-- docs/
|   |-- whitepaper.md
|   |-- model-rubric.md
|   |-- numerical-framework.md
|   `-- validation-agenda.md
|
|-- core_model/
|   |-- __init__.py
|   `-- numerical_framework.py
|
|-- tests/
|   `-- test_numerical_framework.py
|
|-- evaluation/
|   |-- critical_review.md
|   |-- market_landscape.md
|   |-- refresh_plan.md
|   `-- technical_health_check.md
|
`-- archive/
    `-- exploratory-prototype/
        `-- README.md
```

Also update the prototype paragraph to say the older exploratory code is archived and the active implementation is the numerical framework module.

- [ ] **Step 2: Update numerical framework doc**

In `docs/numerical-framework.md`, add a short implementation note after Method Overview:

```markdown
## Reference Implementation

The first reference implementation lives in `core_model/numerical_framework.py`. It implements Monte Carlo uncertainty propagation over a Markov reliance workflow model.

This is not strict Markov chain Monte Carlo. The Markov layer models workflow transitions, while Monte Carlo sampling propagates uncertainty in assumptions and records outcome distributions. Full MCMC posterior inference is a possible later calibration method if real-world evidence requires it.
```

- [ ] **Step 3: Run full verification**

Run:

```bash
.venv/bin/python -m pytest -v
rg -n "scientifically validated|objective detector|guarantees safety|safe to use|strict MCMC" README.md docs core_model tests archive || true
git status --short --branch
```

Expected:

- pytest passes.
- no misleading validation or safety claims appear.
- `strict MCMC` appears only where it is negated or clarified.
- git status shows only intended documentation changes.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md docs/numerical-framework.md
git commit -m "docs: document numerical framework implementation"
```

---

## Final Verification

- [ ] **Step 1: Run tests**

```bash
.venv/bin/python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Inspect active exports**

```bash
.venv/bin/python - <<'PY'
import core_model

print(core_model.__all__)
PY
```

Expected: output includes the numerical framework API and does not include `expertise_required` or `domain_profiles`.

- [ ] **Step 3: Inspect repository shape**

```bash
find . -maxdepth 2 -type d | sort
git status --short --branch
```

Expected:

- active directories include `core_model`, `docs`, `evaluation`, `tests`, and `archive`.
- old prototype directories appear only under `archive/exploratory-prototype`.
- git status is clean after final commit.
