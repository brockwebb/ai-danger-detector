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
