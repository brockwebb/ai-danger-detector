# ADD Baseline Rubric Scorer Design

Date: 2026-05-24

## Purpose

ADD now has evidence records, source metadata, corpus loading, an adjudication protocol, and an illustrative example corpus. The next backend slice is a transparent baseline scorer that converts an evidence unit into a provisional oversight score and oversight band.

This scorer is an assumption-driven reference model, not a calibrated predictor. Its value is that it creates a concrete, inspectable baseline that later Bayesian, Markov, statistical, and ensemble models can challenge or improve.

## Scope

Create `core_model/rubric_scorer.py` with:

- `RubricScoringConfig`: weights and band thresholds.
- `RubricScore`: score, band, factor scores, drivers, and assumptions.
- `score_evidence_unit(unit, config=DEFAULT_RUBRIC_CONFIG)`: score an `EvidenceUnit`.
- `score_feature_row(row, config=DEFAULT_RUBRIC_CONFIG)`: score a row emitted by `EvidenceCorpus.feature_rows()`.

Update package exports, tests, README, and rubric documentation.

## Scoring Inputs

The baseline scorer uses fields already present in `EvidenceUnit`:

- `harm_severity`
- `detectability`
- `reversibility`
- `verification_burden`
- `user_expertise`
- `governance_context`
- `source_quality`

The scorer converts protective factors into risk factors:

- low detectability = `1 - detectability`
- low reversibility = `1 - reversibility`
- user expertise risk is higher for non-experts and lower for experts
- governance risk is inferred conservatively from governance-context text
- source uncertainty risk is higher for weaker source-quality tiers

## Default Weights

Weights should sum to 1.0:

| Factor | Weight |
| --- | --- |
| harm severity | 0.24 |
| low detectability | 0.16 |
| low reversibility | 0.14 |
| verification burden | 0.18 |
| user expertise risk | 0.12 |
| governance risk | 0.10 |
| source uncertainty risk | 0.06 |

These weights are provisional. They should be versioned and revised when validation evidence exists.

## Band Thresholds

The scorer outputs existing `OversightLabel` values:

| Score range | Band |
| --- | --- |
| 0.00 to <0.20 | `casual_exploratory` |
| 0.20 to <0.40 | `assisted_bounded` |
| 0.40 to <0.60 | `trained_review_required` |
| 0.60 to <0.80 | `expert_review_required` |
| 0.80 to 1.00 | `expert_led_or_no_autonomous_use` |

These thresholds are ordinal cut points, not validated probability thresholds.

## Conservative Escalation Rules

The weighted score is the baseline, but a few conservative rules prevent obviously high-risk combinations from being under-banded:

- harm severity >= 0.80 and detectability <= 0.35 escalates to at least expert review;
- harm severity >= 0.80 and reversibility <= 0.35 escalates to at least expert review;
- verification burden >= 0.80 and user expertise is non-expert or unknown escalates to at least expert review;
- harm severity >= 0.90, detectability <= 0.30, and reversibility <= 0.30 escalates to expert-led or no autonomous use.

The output should include drivers explaining any escalation.

## Drivers and Assumptions

`RubricScore` should preserve interpretability:

- `factor_scores`: normalized factor values used by the weighted score.
- `drivers`: high factor names and triggered escalation rules.
- `assumptions`: plain-language notes that the score is provisional and that governance/source heuristics are simple.

## Testing

Tests live in `tests/test_rubric_scorer.py`.

Required coverage:

- Low-consequence, high-detectability examples score into a low oversight band.
- Increasing harm severity increases score.
- Lower detectability increases score.
- Lower source quality increases score or preserves band while adding uncertainty driver.
- High harm plus low detectability triggers conservative escalation.
- Feature rows from `EvidenceCorpus.feature_rows()` can be scored.
- The synthetic example corpus can be scored, and high-stakes examples produce higher bands than creative/education examples.

## Non-Goals

- No training.
- No Bayesian updating.
- No model performance claims.
- No GUI.
- No automatic natural-language parsing.

## Success Criteria

- A caller can score an `EvidenceUnit` or feature row.
- The output is inspectable and includes drivers.
- Monotonicity tests protect the main risk relationships.
- The example corpus can produce baseline scores.
- Full test suite passes.
