# Evidence-to-Bayesian Calibration Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert eligible `EvidenceCorpus` records into traceable `BetaObservation` objects for bounded Bayesian calibration parameters.

**Architecture:** Add a standalone `core_model/calibration_observations.py` module that depends on the evidence corpus/schema and beta-binomial primitive, but not on scorers or model comparison. The module returns observation sets with explicit exclusions so data gaps are visible instead of hidden.

**Tech Stack:** Python 3.12-3.14, dataclasses, enums, pytest.

---

## File Structure

- Create `core_model/calibration_observations.py`: parameter enum, config, exclusion records, observation-set result, and corpus-to-observation builder.
- Modify `core_model/__init__.py`: export the public calibration bridge API.
- Create `tests/test_calibration_observations.py`: TDD coverage for mappings, exclusions, weighting, summaries, validation, and exports.
- Modify `README.md`: list the new bridge module and tests.
- Modify `docs/evidence-data-architecture.md`: document the bridge as evidence-to-observation machinery.
- Modify `docs/numerical-framework.md`: document pseudo-observations and the separation from posterior updates.
- Modify `docs/validation-agenda.md`: name bridge mappings as assumptions needing sensitivity checks.

---

### Task 1: Calibration Observation Bridge

**Files:**
- Create: `tests/test_calibration_observations.py`
- Create: `core_model/calibration_observations.py`
- Modify: `core_model/__init__.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_calibration_observations.py` with helper fixtures and tests for:

- `p_error_per_task` mapping from outcome labels.
- Exclusion of unknown/unresolved outcomes.
- Detectability and reversibility pseudo-observations.
- Confidence and quality weighting.
- Explicit experimental-source inclusion.
- Empty observation-set summaries.
- Config validation and unsupported parameters.
- Public API exports.

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_calibration_observations.py -v
```

Expected: FAIL during import because `core_model.calibration_observations` does not exist.

- [ ] **Step 3: Implement `core_model/calibration_observations.py`**

Add:

```python
class CalibrationParameter(str, Enum):
    P_ERROR_PER_TASK = "p_error_per_task"
    DETECTABILITY = "detectability"
    REVERSIBILITY = "reversibility"
```

Add frozen dataclasses:

- `CalibrationObservationConfig`
- `ExcludedEvidence`
- `CalibrationObservationSet`

Add:

```python
def build_calibration_observations(
    corpus: EvidenceCorpus,
    parameter: CalibrationParameter | str,
    *,
    config: CalibrationObservationConfig | None = None,
) -> CalibrationObservationSet:
    ...
```

Core rules:

- Unsupported parameter raises `ValueError`.
- Experimental sources are excluded unless `include_experimental_sources=True`.
- Quarantined, deprecated, and removed sources are excluded.
- Record-level quarantined evidence is excluded.
- Records below `minimum_confidence` are excluded.
- `p_error_per_task` maps harm/loss/near-miss/corrected-error to event observations and benign use to non-event observations.
- Detectability and reversibility use pseudo-observation counts based on scalar field values.
- Observation weight is `quality_weight[source_quality] * confidence`.
- Empty observation sets return normally with exclusions.

- [ ] **Step 4: Export the public API**

Update `core_model/__init__.py` to import and expose:

- `CalibrationObservationConfig`
- `CalibrationObservationSet`
- `CalibrationParameter`
- `ExcludedEvidence`
- `build_calibration_observations`

- [ ] **Step 5: Verify bridge tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_calibration_observations.py -v
```

Expected: PASS.

- [ ] **Step 6: Verify targeted regression**

Run:

```bash
.venv/bin/python -m pytest tests/test_calibration_observations.py tests/test_bayesian_calibration.py tests/test_evidence_corpus.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit code**

Run:

```bash
git add core_model/calibration_observations.py core_model/__init__.py tests/test_calibration_observations.py
git commit -m "feat: add evidence calibration bridge"
```

---

### Task 2: Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/evidence-data-architecture.md`
- Modify: `docs/numerical-framework.md`
- Modify: `docs/validation-agenda.md`

- [ ] **Step 1: Update README**

Mention evidence-to-calibration observation bridging in the implementation summary, add `calibration_observations.py` to the `core_model/` tree, and add `test_calibration_observations.py` to the `tests/` tree.

- [ ] **Step 2: Update framework docs**

Document that the bridge turns eligible evidence into traceable beta observations, with scalar detectability/reversibility represented as pseudo-observations and posterior updates kept separate.

- [ ] **Step 3: Update validation docs**

Document that bridge mappings and weights are assumptions requiring sensitivity checks.

- [ ] **Step 4: Full verification and guard scans**

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

Expected: tests pass, guard scans are quiet except intentional historical artifacts if any, and git status only shows intended docs.

- [ ] **Step 5: Commit docs**

Run:

```bash
git add README.md docs/evidence-data-architecture.md docs/numerical-framework.md docs/validation-agenda.md
git commit -m "docs: document evidence calibration bridge"
```

---

### Task 3: Final Integration

**Files:**
- No new files.

- [ ] **Step 1: Final verification**

Run:

```bash
.venv/bin/python -m pytest -v
.venv/bin/python - <<'PY'
from core_model import (
    CalibrationObservationConfig,
    CalibrationParameter,
    build_calibration_observations,
)

print(CalibrationObservationConfig().summary())
print(CalibrationParameter.P_ERROR_PER_TASK.value)
print(callable(build_calibration_observations))
PY
git status --short --branch
```

Expected: tests pass, imports work, and branch is clean.

- [ ] **Step 2: Merge and push after verification**

Fast-forward `main`, verify tests on `main`, delete the local feature branch, and push `main`.
