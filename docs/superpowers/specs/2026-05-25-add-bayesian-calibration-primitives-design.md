# ADD Bayesian Calibration Primitives Design

Date: 2026-05-25

## Purpose

ADD needs Bayesian machinery before it can honestly add Bayesian scorers or probability-native metrics. This slice adds small, auditable beta-binomial calibration primitives for bounded probabilities such as `p_error_per_task`, detectability, reversibility, and threshold-crossing rates.

The goal is not to claim empirical calibration. The goal is to create explicit objects for priors, observations, posteriors, evidence traceability, and uncertainty summaries so later model work has a disciplined foundation.

## Scope

Create `core_model/bayesian_calibration.py` with:

- `BetaPrior`: prior alpha, beta, parameter name, version, and notes.
- `BetaObservation`: successes, failures, optional weight, source ID, evidence IDs, and notes.
- `BetaPosterior`: posterior alpha, beta, original prior, observations, effective sample size, source IDs, evidence IDs, mean, variance, and summary.
- `update_beta_binomial(prior, observations)`: weighted beta-binomial update.

Update exports, tests, README, evidence architecture, numerical framework, and validation agenda.

## Interpretation

The primitive models bounded event probabilities. Examples:

- error probability per bounded task trial,
- probability a relevant reviewer detects an issue,
- probability a harm can be reversed,
- probability a case crosses a review threshold.

It does not model unbounded rates or event counts over exposure. Parameters such as `lambda_error_rate` still need Gamma-Poisson or related count-model machinery in a later slice.

## Weighting

Observations should support a non-negative `weight` so weak, synthetic, or low-quality evidence can contribute less than adjudicated evidence without pretending it is absent. Weighted counts should update as:

```text
posterior_alpha = prior_alpha + sum(successes * weight)
posterior_beta = prior_beta + sum(failures * weight)
effective_sample_size = sum((successes + failures) * weight)
```

The implementation should reject negative weights, negative counts, and priors with non-positive alpha or beta.

## Traceability

Each posterior should preserve:

- the original prior,
- the exact observations used,
- sorted source IDs,
- sorted evidence IDs,
- effective sample size,
- observation notes.

This supports TEVV review and later source sensitivity analysis.

## Uncertainty Summary

`BetaPosterior.summary()` should return a plain dictionary suitable for logs or documentation:

- parameter name,
- prior version,
- alpha,
- beta,
- mean,
- variance,
- effective sample size,
- source IDs,
- evidence IDs,
- notes.

No credible interval is required in this slice. That can be added later when the project has a clearer reporting convention for posterior intervals.

## Non-Goals

- No Bayesian scorer.
- No MCMC.
- No Gamma-Poisson or count-rate model.
- No probability-native model comparison metrics yet.
- No claim that any existing evidence has calibrated ADD.

## Tests

Add tests that prove:

- priors reject non-positive alpha or beta,
- observations reject negative successes, failures, or weights,
- unweighted observations update alpha and beta correctly,
- weighted observations update alpha, beta, and effective sample size correctly,
- posterior mean and variance match beta distribution formulas,
- posterior traceability preserves source IDs, evidence IDs, and notes,
- public API exports are present.

## Success Criteria

- ADD has tested beta-binomial primitives for bounded probability calibration.
- The primitives preserve evidence/source traceability.
- The implementation distinguishes bounded probabilities from unbounded rates.
- Documentation warns that this is calibration machinery, not a calibrated model.
- Full test suite passes.
