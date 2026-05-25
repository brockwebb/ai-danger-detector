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


@dataclass(frozen=True)
class EvaluationRow:
    evidence_id: str
    source_id: str
    domain: str
    task_type: str
    adjudicated_band: OversightLabel
    predicted_band: OversightLabel
    adjudicated_ordinal: int | None
    predicted_ordinal: int
    score: float
    drivers: tuple[str, ...]
    band_error: int | None
    error_direction: str
    is_evaluable: bool


@dataclass(frozen=True)
class EvaluationReport:
    rows: tuple[EvaluationRow, ...]
    metrics: dict[str, float]
    predicted_band_counts: dict[str, int]
    adjudicated_band_counts: dict[str, int]
    coverage_summary: dict[str, dict[str, int]]
    record_count: int
    evaluable_count: int


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _band_error_direction(band_error: int | None) -> str:
    if band_error is None:
        return "not_evaluable"
    if band_error == 0:
        return "match"
    if band_error < 0:
        return "under_escalation"
    return "over_escalation"


def _make_row(unit: EvidenceUnit, score: RubricScore) -> EvaluationRow:
    adjudicated_ordinal = BAND_ORDINALS.get(unit.oversight_label)
    predicted_ordinal = BAND_ORDINALS[score.band]
    band_error = (
        None
        if adjudicated_ordinal is None
        else predicted_ordinal - adjudicated_ordinal
    )
    return EvaluationRow(
        evidence_id=unit.evidence_id,
        source_id=unit.source_id,
        domain=unit.domain,
        task_type=unit.task_type,
        adjudicated_band=unit.oversight_label,
        predicted_band=score.band,
        adjudicated_ordinal=adjudicated_ordinal,
        predicted_ordinal=predicted_ordinal,
        score=score.score,
        drivers=score.drivers,
        band_error=band_error,
        error_direction=_band_error_direction(band_error),
        is_evaluable=adjudicated_ordinal is not None,
    )


def _metrics(rows: tuple[EvaluationRow, ...], threshold_band: OversightLabel) -> dict[str, float]:
    evaluable = tuple(row for row in rows if row.is_evaluable)
    evaluable_count = len(evaluable)
    exact_matches = sum(1 for row in evaluable if row.band_error == 0)
    absolute_error = sum(abs(row.band_error or 0) for row in evaluable)
    under_escalations = sum(1 for row in evaluable if (row.band_error or 0) < 0)
    over_escalations = sum(1 for row in evaluable if (row.band_error or 0) > 0)

    threshold = BAND_ORDINALS[threshold_band]
    threshold_positive = tuple(
        row for row in evaluable if (row.adjudicated_ordinal or 0) >= threshold
    )
    threshold_negative = tuple(
        row for row in evaluable if (row.adjudicated_ordinal or 0) < threshold
    )
    false_reassurances = sum(
        1 for row in threshold_positive if row.predicted_ordinal < threshold
    )
    false_escalations = sum(
        1 for row in threshold_negative if row.predicted_ordinal >= threshold
    )

    return {
        "exact_band_agreement": _safe_rate(exact_matches, evaluable_count),
        "mean_absolute_band_error": _safe_rate(absolute_error, evaluable_count),
        "under_escalation_rate": _safe_rate(under_escalations, evaluable_count),
        "over_escalation_rate": _safe_rate(over_escalations, evaluable_count),
        "false_reassurance_rate": _safe_rate(
            false_reassurances, len(threshold_positive)
        ),
        "false_escalation_rate": _safe_rate(false_escalations, len(threshold_negative)),
    }


def evaluate_corpus(
    corpus: EvidenceCorpus,
    *,
    scorer: Callable[[EvidenceUnit], RubricScore] = score_evidence_unit,
    threshold_band: OversightLabel = OversightLabel.TRAINED_REVIEW_REQUIRED,
) -> EvaluationReport:
    rows = tuple(_make_row(unit, scorer(unit)) for unit in corpus.evidence)
    return EvaluationReport(
        rows=rows,
        metrics=_metrics(rows, threshold_band),
        predicted_band_counts=dict(
            sorted(Counter(row.predicted_band.value for row in rows).items())
        ),
        adjudicated_band_counts=dict(
            sorted(Counter(row.adjudicated_band.value for row in rows).items())
        ),
        coverage_summary=corpus.coverage_summary(),
        record_count=len(rows),
        evaluable_count=sum(1 for row in rows if row.is_evaluable),
    )
