import pytest

from core_model.numerical_framework import (
    MarkovState,
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
