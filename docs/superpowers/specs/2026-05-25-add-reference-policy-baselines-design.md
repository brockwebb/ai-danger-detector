# ADD Reference Policy Baselines Design

Date: 2026-05-25

## Purpose

ADD now has an ordinal evaluation runner and a model-comparison layer, but it still has only one substantive registered scorer: the provisional rubric baseline. The next TEVV slice should add deterministic reference policy baselines so every future model can be compared against simple, transparent alternatives.

These baselines are not candidate risk models. They are test controls. They answer questions such as: does a proposed scorer beat always requiring trained review, always requiring expert review, or always allowing casual exploratory use? Do false reassurance and over-escalation move in expected directions?

## Scope

Create `core_model/reference_scorers.py` with deterministic `ScorerDefinition` factories:

- `always_casual_scorer()`
- `always_assisted_scorer()`
- `always_trained_review_scorer()`
- `always_expert_review_scorer()`
- `always_expert_led_scorer()`
- `reference_policy_scorers()`

Each scorer should return a `RubricScore`-compatible object with a fixed oversight band. Each scorer should be registered as `ScorerOutputType.ORDINAL_BAND` so it can run through the existing `compare_models` path.

## Policy Meaning

The reference policies should be named and documented as policies, not learned models:

- `always_casual`: minimum-friction policy; expected to under-escalate serious cases.
- `always_assisted`: bounded-assistance default.
- `always_trained_review`: moderate oversight default.
- `always_expert_review`: conservative review default.
- `always_expert_led`: maximum restriction default; expected to over-escalate low-concern cases.

The descriptions and metric notes should make clear that these are control baselines for TEVV, not recommendations for deployment.

## Metric Compatibility

Reference policy scorers produce ordinal oversight bands. They support the common ordinal and threshold metrics already emitted by `evaluate_corpus` and `compare_models`:

- exact band agreement,
- mean absolute band error,
- under-escalation rate,
- over-escalation rate,
- false reassurance rate,
- false escalation rate.

They do not emit calibrated probabilities. Brier score, log loss, and expected calibration error should remain deferred for these scorers.

## Integration

Update `core_model/__init__.py` to export the reference scorer factories. Tests should compare reference policies and the baseline rubric scorer together using the existing synthetic example corpus and small in-memory corpora.

The model-comparison layer should not need structural changes. If it cannot accept these scorers cleanly, the scorer metadata boundary is wrong and should be fixed narrowly.

## Non-Goals

- No Bayesian scorer.
- No ensemble weighting.
- No probability output.
- No learned parameters.
- No claim that a fixed policy is operationally appropriate.

## Documentation Updates

Update:

- `README.md` to list `reference_scorers.py` and its tests.
- `docs/evidence-data-architecture.md` to describe reference policies as TEVV controls.
- `docs/validation-agenda.md` to state that candidate models should be compared against trivial fixed policies before stronger claims are made.

## Tests

Add tests that prove:

- each fixed-policy scorer returns its intended oversight band,
- all fixed-policy scorers register as ordinal policy baselines with deferred probability metrics,
- `reference_policy_scorers()` returns the expected stable order,
- the policies run through `compare_models`,
- low policies show under-escalation on high-label cases,
- high policies show over-escalation on low-label cases,
- public API exports are present.

## Success Criteria

- ADD has deterministic reference policies available through the same comparison harness as the rubric baseline.
- Reference policies are clearly marked as TEVV controls rather than real risk models.
- False reassurance and over-escalation tradeoffs are visible in tests.
- Full test suite passes.
