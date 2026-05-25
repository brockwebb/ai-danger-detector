# ADD Evaluation Runner Design

Date: 2026-05-25

## Purpose

ADD now has evidence records, an example corpus, a provisional baseline rubric scorer, and basic performance metrics. The next slice adds an evaluation runner that scores a corpus and summarizes how predicted oversight bands compare with adjudicated oversight labels.

This runner is an ordinal evaluation tool. It does not treat the baseline rubric score as a calibrated probability. Probability metrics such as Brier score and log loss should wait until a model explicitly emits calibrated probabilities.

## Scope

Create `core_model/evaluation_runner.py` with:

- `EvaluationRow`: per-record traceability, adjudicated label, predicted band, ordinal band numbers, score, drivers, and error direction.
- `EvaluationReport`: scored rows plus aggregate metrics and coverage summary.
- `evaluate_corpus(corpus, scorer=score_evidence_unit, threshold_band=OversightLabel.TRAINED_REVIEW_REQUIRED)`: score all records in an `EvidenceCorpus`.

Update exports, tests, README, evidence architecture, and validation agenda.

## Metrics

The first report should include:

- record count,
- exact band agreement,
- mean absolute band error,
- under-escalation rate,
- over-escalation rate,
- false reassurance rate at a configurable ordinal threshold,
- false escalation rate at a configurable ordinal threshold,
- predicted band counts,
- adjudicated band counts,
- coverage summary from the corpus.

False reassurance means the adjudicated label is at or above the threshold, but the predicted band is below it. False escalation means the adjudicated label is below the threshold, but the predicted band is at or above it.

## Ordinal Band Mapping

The runner maps `OversightLabel` to ordered integers:

| Label | Ordinal |
| --- | --- |
| `casual_exploratory` | 1 |
| `assisted_bounded` | 2 |
| `trained_review_required` | 3 |
| `expert_review_required` | 4 |
| `expert_led_or_no_autonomous_use` | 5 |

Records with `unknown` adjudicated labels should be excluded from aggregate error-rate metrics but retained in rows with `is_evaluable=False`.

## Row Traceability

Each row should preserve:

- evidence ID,
- source ID,
- domain,
- task type,
- adjudicated oversight label,
- predicted oversight band,
- score,
- scorer drivers,
- ordinal band error when evaluable,
- error direction: `match`, `under_escalation`, `over_escalation`, or `not_evaluable`.

## Non-Goals

- No probability calibration metrics in this slice.
- No command-line interface.
- No file output writer.
- No model training or Bayesian updating.
- No GUI.

## Success Criteria

- A caller can evaluate an in-memory `EvidenceCorpus`.
- Rows preserve traceability and scorer drivers.
- Aggregate metrics identify agreement, band error, under-escalation, over-escalation, false reassurance, and false escalation.
- The synthetic example corpus can be evaluated end-to-end.
- Full test suite passes.
