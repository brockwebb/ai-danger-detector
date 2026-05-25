# ADD Model Comparison Metrics Design

Date: 2026-05-25

## Purpose

ADD needs model comparison as part of its test, evaluation, verification, and validation (TEVV) loop. The current evaluation runner can score one model-like callable against adjudicated oversight bands. The next slice should make that comparison explicit: run multiple named scoring approaches on the same evidence corpus, report comparable decision-layer metrics, and preserve model-native metrics where they are statistically appropriate.

This slice should not pretend that all model outputs are interchangeable. A rubric score, an ordinal band, a calibrated probability, a posterior distribution, a Markov workflow outcome, and an ensemble score can all support oversight triage, but they do not justify the same metrics.

The design follows the principle reflected in scikit-learn's model evaluation guidance: choose scoring functions that match the prediction target and the downstream decision. Classification probabilities support proper probability scores such as Brier score or log loss; point decisions support confusion-matrix-derived decision metrics; ordinal predictions support ordinal error metrics. Reference: https://scikit-learn.org/stable/modules/model_evaluation.html

## Scope

Create a model-comparison layer that can:

- register named scorers with metadata about their output type,
- evaluate each scorer through the existing ordinal `evaluate_corpus` path when it emits an oversight band,
- report one `EvaluationReport` per scorer,
- summarize common TEVV decision metrics across scorers,
- record which model-native metric families are applicable, deferred, or unavailable.

The initial implementation should stay standard-library only and should reuse the current baseline rubric scorer as the first scorer.

## Metric Layers

The comparison report should separate three layers:

1. **Common decision-layer metrics.** These compare each scorer after it has been mapped to the shared oversight-band decision target. The first common metrics are exact band agreement, mean absolute band error, under-escalation rate, over-escalation rate, false reassurance rate, and false escalation rate.
2. **Model-native metrics.** These are valid only when the scorer output supports them. A calibrated probability model may support Brier score, log loss, expected calibration error, or probability calibration plots. An ordinal model supports band-error metrics. A Markov workflow model may support workflow-outcome rates, transition fit, and path likelihood if transition evidence exists.
3. **Equivalence notes.** These explain how a model output was mapped into an oversight band and what information was lost. For example, a probability threshold can become an oversight band for decision comparison, but ordinal band agreement does not test whether the probability was calibrated.

## Scorer Definition

Add a small dataclass for scorer metadata:

- `name`: stable identifier such as `baseline_rubric`.
- `description`: short human-readable description.
- `output_type`: first values should include `ordinal_band`, `probability`, `distribution`, `workflow`, and `ensemble`.
- `scorer`: callable that accepts an `EvidenceUnit` and returns a score object compatible with `evaluate_corpus`.
- `native_metric_notes`: tuple of strings naming applicable or deferred metric families.
- `equivalence_notes`: tuple of strings explaining how the output maps to oversight-band comparison.

The initial registry can expose `baseline_rubric_scorer()`, returning one `ScorerDefinition` for `score_evidence_unit`.

## Comparison Report

Add `core_model/model_comparison.py` with:

- `ScorerDefinition`,
- `ModelComparisonRow`,
- `ModelComparisonReport`,
- `baseline_rubric_scorer()`,
- `compare_models(corpus, scorers, threshold_band=OversightLabel.TRAINED_REVIEW_REQUIRED)`.

`ModelComparisonRow` should include:

- scorer name,
- output type,
- evaluation report,
- common metrics copied from the evaluation report,
- native metric notes,
- equivalence notes.

`ModelComparisonReport` should include:

- rows,
- scorer count,
- record count,
- coverage summary,
- `best_by_metric(metric_name, lower_is_better=False)` helper for simple inspection.

The helper should be deliberately modest. It can identify the best available scorer for a named numeric metric, but it should not collapse all metrics into a single rank.

## Metric Compatibility Rules

The implementation should make compatibility explicit:

- Every scorer in this slice must return an oversight band so it can enter the common ordinal comparison.
- Probability-only metrics must not be computed unless a scorer explicitly provides calibrated probabilities.
- Brier score, log loss, and expected calibration error remain deferred for the baseline rubric scorer because its score is not a calibrated probability.
- Missing or inapplicable native metrics should be represented with notes rather than fake numeric values.

## Non-Goals

- No Bayesian posterior implementation in this slice.
- No ensemble weighting in this slice.
- No model fitting, training, cross-validation, or hyperparameter search.
- No scikit-learn dependency.
- No probability calibration claims for the baseline rubric scorer.
- No global winner score that hides metric tradeoffs.

## Documentation Updates

Update:

- `README.md` to list the model-comparison module and tests.
- `docs/evidence-data-architecture.md` to explain the metric-layer split.
- `docs/validation-agenda.md` to name model comparison as part of the TEVV loop and warn against incompatible metric comparisons.

## Tests

Add tests that prove:

- the baseline scorer can be registered with metric compatibility notes,
- `compare_models` evaluates the example corpus with the baseline scorer,
- comparison rows preserve per-model reports and common metrics,
- `best_by_metric` selects by explicit metric only,
- probability metrics are not emitted for the baseline rubric scorer.

## Success Criteria

- ADD can compare one or more named scorers on the same corpus.
- Common TEVV decision metrics are available for each scorer.
- Model-native metric applicability is documented in machine-readable fields.
- Incompatible metrics are deferred rather than fabricated.
- Full test suite passes.
