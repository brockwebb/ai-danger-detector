"""Active AI Danger Detector backend API."""

from .evidence_schema import (
    EvidenceQualityTier,
    EvidenceType,
    EvidenceUnit,
    OutcomeLabel,
    OversightLabel,
    UserExpertise,
)
from .evidence_corpus import DataSnapshot, EvidenceCorpus, EvidenceSplit
from .evidence_io import (
    EvidenceLoadError,
    load_corpus,
    load_evidence_units,
    load_source_registry,
)

from .numerical_framework import (
    DistributionSpec,
    MarkovState,
    ScenarioSpec,
    SampledScenario,
    SimulationConfig,
    SimulationResult,
    WorkflowResult,
    build_transition_matrix,
    calculate_oversight_score,
    evaluate_markov_workflow,
    run_monte_carlo,
    summarize_simulation,
    validate_transition_matrix,
)
from .performance_metrics import (
    brier_score,
    expected_calibration_error,
    false_escalation_rate,
    false_reassurance_rate,
    interval_coverage,
    log_loss,
)
from .source_registry import (
    EvidenceSource,
    SourceAuditEvent,
    SourceRegistry,
    SourceStatus,
    SourceType,
)

__all__ = [
    "DistributionSpec",
    "DataSnapshot",
    "EvidenceSource",
    "EvidenceCorpus",
    "EvidenceLoadError",
    "EvidenceSplit",
    "EvidenceQualityTier",
    "EvidenceType",
    "EvidenceUnit",
    "MarkovState",
    "OutcomeLabel",
    "OversightLabel",
    "ScenarioSpec",
    "SampledScenario",
    "SimulationConfig",
    "SimulationResult",
    "SourceAuditEvent",
    "SourceRegistry",
    "SourceStatus",
    "SourceType",
    "UserExpertise",
    "WorkflowResult",
    "build_transition_matrix",
    "brier_score",
    "calculate_oversight_score",
    "evaluate_markov_workflow",
    "expected_calibration_error",
    "false_escalation_rate",
    "false_reassurance_rate",
    "interval_coverage",
    "log_loss",
    "load_corpus",
    "load_evidence_units",
    "load_source_registry",
    "run_monte_carlo",
    "summarize_simulation",
    "validate_transition_matrix",
]
