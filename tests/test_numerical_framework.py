import pytest
import numpy as np

from core_model.numerical_framework import (
    DistributionSpec,
    MarkovState,
    ScenarioSpec,
    SimulationConfig,
    build_transition_matrix,
    evaluate_markov_workflow,
    run_monte_carlo,
    summarize_simulation,
    validate_transition_matrix,
)


def _valid_matrix():
    return {
        MarkovState.S0: {MarkovState.S1: 1.0},
        MarkovState.S1: {
            MarkovState.S2: 0.2,
            MarkovState.S3: 0.4,
            MarkovState.S4: 0.4,
        },
        MarkovState.S2: {MarkovState.S6: 1.0},
        MarkovState.S3: {
            MarkovState.S5: 0.5,
            MarkovState.S4: 0.25,
            MarkovState.S6: 0.25,
        },
        MarkovState.S4: {MarkovState.S5: 0.7, MarkovState.S6: 0.3},
        MarkovState.S5: {MarkovState.S7: 0.8, MarkovState.S1: 0.2},
        MarkovState.S6: {MarkovState.S7: 0.9, MarkovState.S8: 0.1},
        MarkovState.S7: {MarkovState.S7: 1.0},
        MarkovState.S8: {MarkovState.S8: 1.0},
    }


def test_all_markov_states_are_named_s0_through_s8():
    assert [state.value for state in MarkovState] == [f"S{i}" for i in range(9)]


def test_validate_transition_matrix_accepts_complete_matrix():
    normalized = validate_transition_matrix(_valid_matrix())

    assert normalized[MarkovState.S0][MarkovState.S1] == pytest.approx(1.0)


def test_validate_transition_matrix_rejects_missing_required_state():
    matrix = _valid_matrix()
    del matrix[MarkovState.S3]

    with pytest.raises(ValueError, match="missing rows"):
        validate_transition_matrix(matrix)


def test_validate_transition_matrix_rejects_negative_probability():
    matrix = _valid_matrix()
    matrix[MarkovState.S1] = {
        MarkovState.S2: -0.1,
        MarkovState.S3: 0.6,
        MarkovState.S4: 0.5,
    }

    with pytest.raises(ValueError, match="negative"):
        validate_transition_matrix(matrix)


def test_validate_transition_matrix_rejects_row_that_does_not_sum_to_one():
    matrix = _valid_matrix()
    matrix[MarkovState.S1] = {
        MarkovState.S2: 0.2,
        MarkovState.S3: 0.2,
        MarkovState.S4: 0.2,
    }

    with pytest.raises(ValueError, match="sum to 1"):
        validate_transition_matrix(matrix)


def test_evaluate_markov_workflow_returns_terminal_probabilities():
    result = evaluate_markov_workflow(_valid_matrix())

    assert result.terminal_probabilities[MarkovState.S7] == pytest.approx(0.9494382022)
    assert result.terminal_probabilities[MarkovState.S8] == pytest.approx(0.0505617978)
    assert sum(result.terminal_probabilities.values()) == pytest.approx(1.0)


def test_evaluate_markov_workflow_tracks_probability_of_unverified_action_path():
    result = evaluate_markov_workflow(_valid_matrix())

    assert result.unverified_action_probability == pytest.approx(0.2247191011)


def test_distribution_spec_samples_fixed_beta_triangular_and_lognormal_values():
    rng = np.random.default_rng(123)

    fixed = DistributionSpec.fixed(0.4).sample(rng, 4)
    beta = DistributionSpec.beta(alpha=2, beta=5).sample(rng, 200)
    triangular = DistributionSpec.triangular(left=0.1, mode=0.3, right=0.9).sample(
        rng, 200
    )
    lognormal = DistributionSpec.lognormal(mean=1.0, sigma=0.25).sample(rng, 200)

    assert fixed.tolist() == [0.4, 0.4, 0.4, 0.4]
    assert beta.min() >= 0.0
    assert beta.max() <= 1.0
    assert triangular.min() >= 0.1
    assert triangular.max() <= 0.9
    assert lognormal.min() > 0.0


def test_build_transition_matrix_uses_scenario_values_to_create_valid_matrix():
    scenario = ScenarioSpec(
        error_probability=DistributionSpec.fixed(0.2),
        severity=DistributionSpec.fixed(0.7),
        detectability=DistributionSpec.fixed(0.6),
        reversibility=DistributionSpec.fixed(0.5),
        verification_burden=DistributionSpec.fixed(0.8),
        governance_strength=DistributionSpec.fixed(0.7),
        conditional_loss=DistributionSpec.fixed(1000.0),
    )

    sampled = scenario.sample(np.random.default_rng(123))
    matrix = build_transition_matrix(sampled)
    normalized = validate_transition_matrix(matrix)

    assert normalized[MarkovState.S0][MarkovState.S1] == pytest.approx(1.0)
    assert normalized[MarkovState.S6][MarkovState.S8] > 0.0


def _scenario(error, severity, detectability, reversibility, burden, governance, loss):
    return ScenarioSpec(
        error_probability=DistributionSpec.fixed(error),
        severity=DistributionSpec.fixed(severity),
        detectability=DistributionSpec.fixed(detectability),
        reversibility=DistributionSpec.fixed(reversibility),
        verification_burden=DistributionSpec.fixed(burden),
        governance_strength=DistributionSpec.fixed(governance),
        conditional_loss=DistributionSpec.fixed(loss),
    )


def test_run_monte_carlo_is_reproducible_with_seed():
    scenario = _scenario(0.2, 0.7, 0.6, 0.5, 0.8, 0.7, 1000.0)
    config = SimulationConfig(iterations=50, seed=123)

    first = run_monte_carlo(scenario, config)
    second = run_monte_carlo(scenario, config)

    assert first.oversight_scores.tolist() == second.oversight_scores.tolist()
    assert (
        first.realized_harm_probabilities.tolist()
        == second.realized_harm_probabilities.tolist()
    )


def test_higher_risk_scenario_increases_expert_led_probability():
    low = run_monte_carlo(
        _scenario(0.03, 0.2, 0.9, 0.9, 0.2, 0.8, 100.0),
        SimulationConfig(iterations=50, seed=123),
    )
    high = run_monte_carlo(
        _scenario(0.5, 0.95, 0.15, 0.1, 0.95, 0.2, 10000.0),
        SimulationConfig(iterations=50, seed=123),
    )

    assert (
        high.threshold_probabilities["expert_led_or_no_autonomous_use"]
        > low.threshold_probabilities["expert_led_or_no_autonomous_use"]
    )
    assert high.expected_losses.mean() > low.expected_losses.mean()


def test_summarize_simulation_reports_documented_outputs():
    result = run_monte_carlo(
        _scenario(0.2, 0.7, 0.6, 0.5, 0.8, 0.7, 1000.0),
        SimulationConfig(iterations=50, seed=123),
    )

    summary = summarize_simulation(result)

    assert set(summary) == {
        "median_oversight_score",
        "oversight_score_interval",
        "p_trained_review_required",
        "p_expert_review_required",
        "p_expert_led_or_no_autonomous_use",
        "p_unverified_action",
        "p_realized_harm",
        "expected_loss",
    }
    assert summary["expected_loss"] > 0.0
