# ADD Evidence Backbone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first backend evidence layer for ADD so evidence records, source metadata, and core performance metrics can be represented and tested.

**Architecture:** Add three focused modules under `core_model/`: `evidence_schema.py` for typed evidence records and quality tiers, `source_registry.py` for source admission/status/audit behavior, and `performance_metrics.py` for calibration-oriented metrics. Keep the implementation pure Python plus NumPy so it fits the current package and remains easy to inspect.

**Tech Stack:** Python 3.12-3.14, dataclasses, enums, standard-library datetime, NumPy, pytest.

---

## File Structure

- Create `core_model/evidence_schema.py`: enums and dataclasses for evidence units, evidence quality, oversight labels, and schema validation.
- Create `core_model/source_registry.py`: source metadata, active/quarantined/deprecated/removed statuses, and audit events.
- Create `core_model/performance_metrics.py`: Brier score, log loss, expected calibration error, false reassurance rate, false escalation rate, and interval coverage.
- Modify `core_model/__init__.py`: export the active evidence backbone API.
- Create `tests/test_evidence_schema.py`: behavior tests for evidence record validation and uncertainty treatment.
- Create `tests/test_source_registry.py`: behavior tests for source lifecycle and audit trail.
- Create `tests/test_performance_metrics.py`: behavior tests for metric calculations.
- Modify `README.md`: note that the backend evidence layer exists after implementation.

---

### Task 1: Evidence Schema

**Files:**
- Create: `tests/test_evidence_schema.py`
- Create: `core_model/evidence_schema.py`
- Modify: `core_model/__init__.py`

- [ ] **Step 1: Write failing tests for evidence records and quality tiers**

Create `tests/test_evidence_schema.py` with:

```python
import pytest

from core_model.evidence_schema import (
    EvidenceQualityTier,
    EvidenceType,
    EvidenceUnit,
    OutcomeLabel,
    OversightLabel,
    UserExpertise,
)


def _valid_evidence(**overrides):
    values = {
        "evidence_id": "case-001",
        "source_id": "src-001",
        "evidence_type": EvidenceType.INCIDENT,
        "collection_date": "2026-05-24",
        "event_date": "2026-05-20",
        "domain": "health",
        "task_type": "medical symptom advice",
        "model_family": "unknown",
        "model_version": "unknown",
        "user_expertise": UserExpertise.NON_EXPERT,
        "governance_context": "informal consumer use",
        "outcome_label": OutcomeLabel.HARM,
        "oversight_label": OversightLabel.EXPERT_REVIEW_REQUIRED,
        "harm_severity": 0.8,
        "detectability": 0.2,
        "reversibility": 0.3,
        "verification_burden": 0.9,
        "workflow_path": ("S0", "S1", "S2", "S6", "S8"),
        "confidence": 0.7,
        "source_quality": EvidenceQualityTier.TIER_3,
        "bias_notes": ("single public report",),
        "relevance_limits": ("consumer health only",),
    }
    values.update(overrides)
    return EvidenceUnit(**values)


def test_evidence_unit_accepts_valid_required_fields():
    unit = _valid_evidence()

    assert unit.evidence_id == "case-001"
    assert unit.is_calibration_eligible is True
    assert unit.uncertainty_multiplier == pytest.approx(1.5)


def test_evidence_unit_rejects_probability_outside_zero_one():
    with pytest.raises(ValueError, match="harm_severity"):
        _valid_evidence(harm_severity=1.5)


def test_evidence_unit_rejects_empty_identifiers():
    with pytest.raises(ValueError, match="evidence_id"):
        _valid_evidence(evidence_id="")


def test_quarantined_evidence_is_not_calibration_eligible():
    unit = _valid_evidence(source_quality=EvidenceQualityTier.QUARANTINED)

    assert unit.is_calibration_eligible is False
    assert unit.uncertainty_multiplier == pytest.approx(3.0)


def test_evidence_unit_serializes_to_model_ready_dict():
    unit = _valid_evidence()

    payload = unit.to_feature_row()

    assert payload["evidence_id"] == "case-001"
    assert payload["evidence_type"] == "incident"
    assert payload["source_quality"] == "tier_3"
    assert payload["harm_severity"] == pytest.approx(0.8)
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_evidence_schema.py -v
```

Expected: FAIL because `core_model.evidence_schema` does not exist.

- [ ] **Step 3: Implement evidence schema**

Create `core_model/evidence_schema.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvidenceType(str, Enum):
    INCIDENT = "incident"
    NEAR_MISS = "near_miss"
    BENIGN_COMPARISON = "benign_comparison"
    BENCHMARK = "benchmark"
    EXPERT_ELICITATION = "expert_elicitation"
    DEPLOYMENT_LOG = "deployment_log"
    USER_STUDY = "user_study"
    CASE_REVIEW = "case_review"
    SYNTHETIC_STRESS_CASE = "synthetic_stress_case"


class OutcomeLabel(str, Enum):
    HARM = "harm"
    LOSS = "loss"
    NEAR_MISS = "near_miss"
    CORRECTED_ERROR = "corrected_error"
    BENIGN_USE = "benign_use"
    UNRESOLVED = "unresolved"
    UNKNOWN = "unknown"


class OversightLabel(str, Enum):
    CASUAL_EXPLORATORY = "casual_exploratory"
    ASSISTED_BOUNDED = "assisted_bounded"
    TRAINED_REVIEW_REQUIRED = "trained_review_required"
    EXPERT_REVIEW_REQUIRED = "expert_review_required"
    EXPERT_LED_OR_NO_AUTONOMOUS_USE = "expert_led_or_no_autonomous_use"
    UNKNOWN = "unknown"


class UserExpertise(str, Enum):
    NON_EXPERT = "non_expert"
    TRAINED = "trained"
    DOMAIN_FAMILIAR = "domain_familiar"
    EXPERT = "expert"
    UNKNOWN = "unknown"


class EvidenceQualityTier(str, Enum):
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4 = "tier_4"
    QUARANTINED = "quarantined"


_UNCERTAINTY_MULTIPLIERS = {
    EvidenceQualityTier.TIER_1: 1.0,
    EvidenceQualityTier.TIER_2: 1.2,
    EvidenceQualityTier.TIER_3: 1.5,
    EvidenceQualityTier.TIER_4: 2.0,
    EvidenceQualityTier.QUARANTINED: 3.0,
}


def _require_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _probability(value: float, field_name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return value


def _as_tuple(values: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    return tuple(str(value).strip() for value in values if str(value).strip())


@dataclass(frozen=True)
class EvidenceUnit:
    evidence_id: str
    source_id: str
    evidence_type: EvidenceType
    collection_date: str
    event_date: str | None
    domain: str
    task_type: str
    model_family: str
    model_version: str
    user_expertise: UserExpertise
    governance_context: str
    outcome_label: OutcomeLabel
    oversight_label: OversightLabel
    harm_severity: float
    detectability: float
    reversibility: float
    verification_burden: float
    workflow_path: tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 0.5
    source_quality: EvidenceQualityTier = EvidenceQualityTier.TIER_3
    bias_notes: tuple[str, ...] = field(default_factory=tuple)
    relevance_limits: tuple[str, ...] = field(default_factory=tuple)
    optional_fields: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _require_text(self.evidence_id, "evidence_id"))
        object.__setattr__(self, "source_id", _require_text(self.source_id, "source_id"))
        object.__setattr__(self, "collection_date", _require_text(self.collection_date, "collection_date"))
        object.__setattr__(self, "domain", _require_text(self.domain, "domain"))
        object.__setattr__(self, "task_type", _require_text(self.task_type, "task_type"))
        object.__setattr__(self, "model_family", _require_text(self.model_family, "model_family"))
        object.__setattr__(self, "model_version", _require_text(self.model_version, "model_version"))
        object.__setattr__(self, "governance_context", _require_text(self.governance_context, "governance_context"))
        object.__setattr__(self, "harm_severity", _probability(self.harm_severity, "harm_severity"))
        object.__setattr__(self, "detectability", _probability(self.detectability, "detectability"))
        object.__setattr__(self, "reversibility", _probability(self.reversibility, "reversibility"))
        object.__setattr__(self, "verification_burden", _probability(self.verification_burden, "verification_burden"))
        object.__setattr__(self, "confidence", _probability(self.confidence, "confidence"))
        object.__setattr__(self, "workflow_path", _as_tuple(self.workflow_path))
        object.__setattr__(self, "bias_notes", _as_tuple(self.bias_notes))
        object.__setattr__(self, "relevance_limits", _as_tuple(self.relevance_limits))

    @property
    def is_calibration_eligible(self) -> bool:
        return self.source_quality is not EvidenceQualityTier.QUARANTINED

    @property
    def uncertainty_multiplier(self) -> float:
        return _UNCERTAINTY_MULTIPLIERS[self.source_quality]

    def to_feature_row(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_id": self.source_id,
            "evidence_type": self.evidence_type.value,
            "collection_date": self.collection_date,
            "event_date": self.event_date,
            "domain": self.domain,
            "task_type": self.task_type,
            "model_family": self.model_family,
            "model_version": self.model_version,
            "user_expertise": self.user_expertise.value,
            "governance_context": self.governance_context,
            "outcome_label": self.outcome_label.value,
            "oversight_label": self.oversight_label.value,
            "harm_severity": self.harm_severity,
            "detectability": self.detectability,
            "reversibility": self.reversibility,
            "verification_burden": self.verification_burden,
            "workflow_path": self.workflow_path,
            "confidence": self.confidence,
            "source_quality": self.source_quality.value,
            "bias_notes": self.bias_notes,
            "relevance_limits": self.relevance_limits,
            **self.optional_fields,
        }
```

Update `core_model/__init__.py` to export `EvidenceQualityTier`, `EvidenceType`, `EvidenceUnit`, `OutcomeLabel`, `OversightLabel`, and `UserExpertise`.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_evidence_schema.py -v
```

Expected: all evidence schema tests pass.

- [ ] **Step 5: Commit evidence schema**

```bash
git add core_model tests
git commit -m "feat: add evidence record schema"
```

---

### Task 2: Source Registry

**Files:**
- Create: `tests/test_source_registry.py`
- Create: `core_model/source_registry.py`
- Modify: `core_model/__init__.py`

- [ ] **Step 1: Write failing source registry tests**

Create `tests/test_source_registry.py` with:

```python
import pytest

from core_model.evidence_schema import EvidenceQualityTier
from core_model.source_registry import (
    SourceRegistry,
    SourceStatus,
    SourceType,
)


def test_source_registry_adds_experimental_source():
    registry = SourceRegistry()

    source = registry.add_source(
        source_id="src-001",
        source_name="Public incident review set",
        source_type=SourceType.INCIDENT_REPOSITORY,
        owner_or_publisher="ADD test",
        license_or_access="public",
        update_cadence="periodic",
        coverage=("health", "law"),
        known_biases=("severity bias",),
        quality_tier=EvidenceQualityTier.TIER_3,
    )

    assert source.status is SourceStatus.EXPERIMENTAL
    assert registry.get_source("src-001") == source
    assert registry.audit_events[-1].action == "add_source"


def test_source_registry_prevents_duplicate_source_ids():
    registry = SourceRegistry()
    registry.add_source(
        source_id="src-001",
        source_name="A",
        source_type=SourceType.CASE_SET,
        owner_or_publisher="ADD",
        license_or_access="private",
        update_cadence="one-time",
        coverage=("software",),
        known_biases=(),
        quality_tier=EvidenceQualityTier.TIER_2,
    )

    with pytest.raises(ValueError, match="already exists"):
        registry.add_source(
            source_id="src-001",
            source_name="B",
            source_type=SourceType.CASE_SET,
            owner_or_publisher="ADD",
            license_or_access="private",
            update_cadence="one-time",
            coverage=("software",),
            known_biases=(),
            quality_tier=EvidenceQualityTier.TIER_2,
        )


def test_source_registry_requires_reason_for_non_active_status_change():
    registry = SourceRegistry()
    registry.add_source(
        source_id="src-001",
        source_name="A",
        source_type=SourceType.CASE_SET,
        owner_or_publisher="ADD",
        license_or_access="private",
        update_cadence="one-time",
        coverage=("software",),
        known_biases=(),
        quality_tier=EvidenceQualityTier.TIER_2,
    )

    with pytest.raises(ValueError, match="reason"):
        registry.update_status("src-001", SourceStatus.QUARANTINED)


def test_source_registry_tracks_status_audit_trail():
    registry = SourceRegistry()
    registry.add_source(
        source_id="src-001",
        source_name="A",
        source_type=SourceType.CASE_SET,
        owner_or_publisher="ADD",
        license_or_access="private",
        update_cadence="one-time",
        coverage=("software",),
        known_biases=(),
        quality_tier=EvidenceQualityTier.TIER_2,
    )

    registry.update_status("src-001", SourceStatus.ACTIVE, reason="passed admission review")
    registry.update_status("src-001", SourceStatus.QUARANTINED, reason="degraded holdout calibration")

    source = registry.get_source("src-001")
    assert source.status is SourceStatus.QUARANTINED
    assert source.removal_reason == "degraded holdout calibration"
    assert [event.action for event in registry.audit_events] == [
        "add_source",
        "update_status",
        "update_status",
    ]
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_source_registry.py -v
```

Expected: FAIL because `core_model.source_registry` does not exist.

- [ ] **Step 3: Implement source registry**

Create `core_model/source_registry.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from .evidence_schema import EvidenceQualityTier


class SourceType(str, Enum):
    INCIDENT_REPOSITORY = "incident_repository"
    BENCHMARK_SUITE = "benchmark_suite"
    EXPERT_PANEL = "expert_panel"
    DEPLOYMENT_LOG = "deployment_log"
    USER_STUDY = "user_study"
    AUDIT = "audit"
    LITERATURE_REVIEW = "literature_review"
    CASE_SET = "case_set"
    SYNTHETIC = "synthetic"


class SourceStatus(str, Enum):
    ACTIVE = "active"
    EXPERIMENTAL = "experimental"
    QUARANTINED = "quarantined"
    DEPRECATED = "deprecated"
    REMOVED = "removed"


@dataclass(frozen=True)
class EvidenceSource:
    source_id: str
    source_name: str
    source_type: SourceType
    owner_or_publisher: str
    license_or_access: str
    update_cadence: str
    coverage: tuple[str, ...]
    known_biases: tuple[str, ...]
    quality_tier: EvidenceQualityTier
    status: SourceStatus = SourceStatus.EXPERIMENTAL
    removal_reason: str | None = None


@dataclass(frozen=True)
class SourceAuditEvent:
    source_id: str
    action: str
    reason: str | None
    status: SourceStatus


def _text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _tuple(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(str(value).strip() for value in values if str(value).strip())


class SourceRegistry:
    def __init__(self) -> None:
        self._sources: dict[str, EvidenceSource] = {}
        self._audit_events: list[SourceAuditEvent] = []

    @property
    def audit_events(self) -> tuple[SourceAuditEvent, ...]:
        return tuple(self._audit_events)

    def add_source(
        self,
        *,
        source_id: str,
        source_name: str,
        source_type: SourceType,
        owner_or_publisher: str,
        license_or_access: str,
        update_cadence: str,
        coverage: tuple[str, ...] | list[str],
        known_biases: tuple[str, ...] | list[str],
        quality_tier: EvidenceQualityTier,
    ) -> EvidenceSource:
        source_id = _text(source_id, "source_id")
        if source_id in self._sources:
            raise ValueError(f"source_id already exists: {source_id}")

        source = EvidenceSource(
            source_id=source_id,
            source_name=_text(source_name, "source_name"),
            source_type=source_type,
            owner_or_publisher=_text(owner_or_publisher, "owner_or_publisher"),
            license_or_access=_text(license_or_access, "license_or_access"),
            update_cadence=_text(update_cadence, "update_cadence"),
            coverage=_tuple(coverage),
            known_biases=_tuple(known_biases),
            quality_tier=quality_tier,
        )
        self._sources[source_id] = source
        self._audit_events.append(SourceAuditEvent(source_id, "add_source", None, source.status))
        return source

    def get_source(self, source_id: str) -> EvidenceSource:
        try:
            return self._sources[source_id]
        except KeyError as exc:
            raise KeyError(f"unknown source_id: {source_id}") from exc

    def update_status(
        self,
        source_id: str,
        status: SourceStatus,
        *,
        reason: str | None = None,
    ) -> EvidenceSource:
        if status in {SourceStatus.QUARANTINED, SourceStatus.DEPRECATED, SourceStatus.REMOVED} and not reason:
            raise ValueError("reason is required when quarantining, deprecating, or removing a source")

        source = self.get_source(source_id)
        updated = replace(
            source,
            status=status,
            removal_reason=reason if status is not SourceStatus.ACTIVE else None,
        )
        self._sources[source_id] = updated
        self._audit_events.append(SourceAuditEvent(source_id, "update_status", reason, status))
        return updated
```

Update `core_model/__init__.py` to export `EvidenceSource`, `SourceAuditEvent`, `SourceRegistry`, `SourceStatus`, and `SourceType`.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_source_registry.py -v
```

Expected: all source registry tests pass.

- [ ] **Step 5: Commit source registry**

```bash
git add core_model tests
git commit -m "feat: add evidence source registry"
```

---

### Task 3: Performance Metrics

**Files:**
- Create: `tests/test_performance_metrics.py`
- Create: `core_model/performance_metrics.py`
- Modify: `core_model/__init__.py`

- [ ] **Step 1: Write failing metric tests**

Create `tests/test_performance_metrics.py` with:

```python
import pytest

from core_model.performance_metrics import (
    brier_score,
    expected_calibration_error,
    false_escalation_rate,
    false_reassurance_rate,
    interval_coverage,
    log_loss,
)


def test_brier_score_returns_mean_squared_probability_error():
    assert brier_score([0.1, 0.8, 0.6], [0, 1, 1]) == pytest.approx(0.07)


def test_log_loss_clips_extreme_probabilities():
    value = log_loss([0.0, 1.0], [0, 1])

    assert value < 1e-6


def test_expected_calibration_error_bins_predictions():
    value = expected_calibration_error(
        probabilities=[0.1, 0.2, 0.8, 0.9],
        outcomes=[0, 0, 1, 1],
        bins=2,
    )

    assert value == pytest.approx(0.15)


def test_false_reassurance_rate_counts_concern_cases_scored_too_low():
    rate = false_reassurance_rate(
        predicted_scores=[0.2, 0.4, 0.9, 0.1],
        true_labels=[1, 1, 1, 0],
        threshold=0.5,
    )

    assert rate == pytest.approx(2 / 3)


def test_false_escalation_rate_counts_low_concern_cases_scored_too_high():
    rate = false_escalation_rate(
        predicted_scores=[0.2, 0.7, 0.8, 0.1],
        true_labels=[0, 0, 1, 0],
        threshold=0.5,
    )

    assert rate == pytest.approx(1 / 3)


def test_interval_coverage_counts_outcomes_inside_interval():
    coverage = interval_coverage(
        lower=[0.1, 0.2, 0.4],
        upper=[0.5, 0.7, 0.9],
        observed=[0.3, 0.8, 0.9],
    )

    assert coverage == pytest.approx(2 / 3)
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_performance_metrics.py -v
```

Expected: FAIL because `core_model.performance_metrics` does not exist.

- [ ] **Step 3: Implement metrics**

Create `core_model/performance_metrics.py` with:

```python
from __future__ import annotations

import numpy as np


def _arrays(predicted, observed) -> tuple[np.ndarray, np.ndarray]:
    predicted_array = np.asarray(predicted, dtype=float)
    observed_array = np.asarray(observed, dtype=float)
    if predicted_array.shape != observed_array.shape:
        raise ValueError("predicted and observed values must have the same shape")
    if predicted_array.size == 0:
        raise ValueError("metric inputs must not be empty")
    return predicted_array, observed_array


def brier_score(probabilities, outcomes) -> float:
    probabilities, outcomes = _arrays(probabilities, outcomes)
    return float(np.mean((probabilities - outcomes) ** 2))


def log_loss(probabilities, outcomes, *, epsilon: float = 1e-15) -> float:
    probabilities, outcomes = _arrays(probabilities, outcomes)
    probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)
    losses = -(outcomes * np.log(probabilities) + (1.0 - outcomes) * np.log(1.0 - probabilities))
    return float(np.mean(losses))


def expected_calibration_error(probabilities, outcomes, *, bins: int = 10) -> float:
    probabilities, outcomes = _arrays(probabilities, outcomes)
    if bins <= 0:
        raise ValueError("bins must be positive")

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = probabilities.size
    error = 0.0

    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not np.any(mask):
            continue
        confidence = float(np.mean(probabilities[mask]))
        accuracy = float(np.mean(outcomes[mask]))
        error += (np.sum(mask) / total) * abs(confidence - accuracy)

    return float(error)


def false_reassurance_rate(predicted_scores, true_labels, *, threshold: float) -> float:
    predicted_scores, true_labels = _arrays(predicted_scores, true_labels)
    positives = true_labels == 1
    if not np.any(positives):
        return 0.0
    return float(np.mean(predicted_scores[positives] < threshold))


def false_escalation_rate(predicted_scores, true_labels, *, threshold: float) -> float:
    predicted_scores, true_labels = _arrays(predicted_scores, true_labels)
    negatives = true_labels == 0
    if not np.any(negatives):
        return 0.0
    return float(np.mean(predicted_scores[negatives] >= threshold))


def interval_coverage(lower, upper, observed) -> float:
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    observed_array = np.asarray(observed, dtype=float)
    if lower_array.shape != upper_array.shape or lower_array.shape != observed_array.shape:
        raise ValueError("lower, upper, and observed values must have the same shape")
    if lower_array.size == 0:
        raise ValueError("metric inputs must not be empty")
    inside = (observed_array >= lower_array) & (observed_array <= upper_array)
    return float(np.mean(inside))
```

Update `core_model/__init__.py` to export `brier_score`, `expected_calibration_error`, `false_escalation_rate`, `false_reassurance_rate`, `interval_coverage`, and `log_loss`.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_performance_metrics.py -v
```

Expected: all performance metric tests pass.

- [ ] **Step 5: Commit performance metrics**

```bash
git add core_model tests
git commit -m "feat: add evidence performance metrics"
```

---

### Task 4: Documentation and Full Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/evidence-data-architecture.md`

- [ ] **Step 1: Update README implementation paragraph**

Update the current implementation paragraph in `README.md` to say:

```markdown
The active Python implementation lives in `core_model/`. It includes the numerical framework plus an evidence backbone for typed evidence records, source registry metadata, and calibration-oriented performance metrics.
```

- [ ] **Step 2: Add implementation note to evidence architecture doc**

Add this section after `Initial Implementation Path` in `docs/evidence-data-architecture.md`:

```markdown
## Reference Implementation

The first backend evidence implementation lives in `core_model/evidence_schema.py`, `core_model/source_registry.py`, and `core_model/performance_metrics.py`.

It does not yet collect real evidence or tune model weights. Its purpose is to define the typed records, source lifecycle behavior, and metric calculations that future calibration runs will need.
```

- [ ] **Step 3: Run full verification**

Run:

```bash
.venv/bin/python -m pytest -v
rg -n "scientifically validated|objective detector|guarantees safety|safe to use|tick|deer|acorn|forest" README.md docs core_model tests || true
git status --short --branch
```

Expected:

- all tests pass,
- no placeholders, overclaims, or stray wrong-project terms appear,
- status shows only intended docs changes before commit.

- [ ] **Step 4: Commit documentation update**

```bash
git add README.md docs/evidence-data-architecture.md
git commit -m "docs: document evidence backbone implementation"
```

---

## Final Verification

- [ ] **Step 1: Run full tests**

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

Expected: output includes evidence schema, source registry, performance metric, and numerical framework exports.

- [ ] **Step 3: Check repository state**

```bash
git status --short --branch
```

Expected: working tree is clean on the evidence implementation branch.
