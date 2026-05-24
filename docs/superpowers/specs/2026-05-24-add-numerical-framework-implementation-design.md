# ADD Numerical Framework Implementation Design

Date: 2026-05-24

## Purpose

Create the first real implementation of the ADD numerical framework described in `docs/numerical-framework.md`.

The goal is a clean reference implementation that technical readers can inspect, test, critique, and extend. It should be explicit about uncertainty and workflow assumptions without claiming empirical validation.

## Scope

This phase has two parts:

1. Archive the older exploratory prototype code so the repository has a cleaner public shape.
2. Add a tested numerical framework implementation centered on Bayesian-style parameter uncertainty, Monte Carlo propagation, and Markov workflow analysis.

The implementation should make the current foundation more credible, but it should not present ADD as a validated decision instrument.

## MCMC Clarification

The first implementation will not claim to use MCMC in the strict Bayesian inference sense.

ADD will use:

- Monte Carlo simulation to sample uncertain parameters and report distributions of outcomes.
- A Markov workflow model to represent movement through AI reliance states.

Together, these mean the simulation repeatedly samples assumptions and evaluates a Markov chain. That is connected and intentional, but it is not the same thing as Markov chain Monte Carlo methods such as Metropolis-Hastings or Hamiltonian Monte Carlo.

MCMC can be a later extension if real calibration data creates posterior distributions that cannot be sampled directly or approximated with simple update rules.

## Repository Shape

Older exploratory work should move under:

`archive/exploratory-prototype/`

That archive should include a README explaining that the contents are historical, exploratory, and not the reference implementation for current use.

Candidate archived material:

- `applications/`
- `data_generation/`
- `machine_learning/`
- `model_validation/`
- `visualization/`
- older prototype modules currently under `core_model/`
- `parameter_sensitivity.png`

The current `docs/`, `evaluation/`, `README.md`, and new reference implementation should remain at the top level.

The active `core_model/` package should become the home of the new numerical framework, not a mix of old and new assumptions.

## Active Implementation

Add `core_model/numerical_framework.py` as the primary reference implementation.

It should provide:

- explicit state labels for `S0` through `S8`,
- dataclasses for distributions, scenario assumptions, simulation configuration, and simulation results,
- validation for Markov transition matrices,
- direct sampling from fixed, beta, triangular, and log-normal distributions,
- Monte Carlo simulation with seeded reproducibility,
- analytical Markov terminal-probability evaluation for each simulation draw,
- oversight score and oversight band assignment,
- summary outputs with medians, uncertainty intervals, threshold probabilities, workflow probabilities, and expected loss.

The implementation should prioritize readability and auditability over cleverness. A reader should be able to trace how an assumption becomes a simulated outcome.

## API Shape

The module should expose a small public API:

- `MarkovState`
- `DistributionSpec`
- `ScenarioSpec`
- `SimulationConfig`
- `SimulationResult`
- `validate_transition_matrix`
- `evaluate_markov_workflow`
- `run_monte_carlo`
- `summarize_simulation`

`core_model/__init__.py` should export the active API and avoid exporting archived prototype functions.

## Modeling Choices

The first scoring model should be deliberately simple:

- high error probability increases oversight need,
- high severity increases oversight need,
- low detectability increases oversight need,
- low reversibility increases oversight need,
- high verification burden increases oversight need,
- governance strength should reduce unchecked reliance paths in the Markov workflow.

This scoring rule is provisional. It exists to make the numerical workflow executable and testable, not to assert calibrated coefficients.

Severity or expected loss should remain separate from the `S8` realized-harm state. `S8` means material harm occurred; conditional loss magnitude is a separate sampled value used in expected loss calculations.

## Tests

Use TDD for implementation. Add `tests/test_numerical_framework.py` before production code.

Tests should cover:

- transition matrix validation rejects missing states, negative probabilities, and rows that do not sum to one,
- terminal Markov probabilities are computed correctly for a simple workflow,
- seeded Monte Carlo runs are reproducible,
- higher-risk assumptions increase oversight threshold probabilities relative to lower-risk assumptions,
- expected loss uses both realized-harm probability and conditional loss magnitude,
- simulation summaries include the outputs promised by `docs/numerical-framework.md`.

## Packaging

Add `pyproject.toml` with:

- `requires-python = ">=3.12,<3.15"`,
- runtime dependencies for the implementation,
- pytest as a development/test dependency.

The local `.venv` already runs Python 3.14.3 and has NumPy, SciPy, pandas, and pytest available. The implementation should use only the dependencies it needs.

## Documentation Updates

Update `README.md` to show the new repository shape:

- active reference implementation in `core_model/`,
- historical exploratory prototype in `archive/exploratory-prototype/`,
- docs as the source of methodological intent.

Update `docs/numerical-framework.md` to mention that a first reference implementation exists and to clarify that the current approach is Monte Carlo over a Markov workflow model, not strict MCMC.

## Non-Goals

This phase will not:

- calibrate the model against real-world case data,
- implement full Bayesian posterior inference with MCMC,
- build a CLI or web app,
- train machine-learning models,
- claim validated predictive accuracy,
- preserve the older prototype as active importable code.

## Success Criteria

The work is successful when:

- older exploratory material is clearly archived,
- the active implementation matches the numerical framework document,
- tests pass under the local Python environment,
- the README no longer makes the old prototype look active,
- the docs accurately describe the Monte Carlo plus Markov approach,
- the result is honest enough for public reference and concrete enough for future contributors.
