from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np


class MarkovState(str, Enum):
    S0 = "S0"
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"
    S5 = "S5"
    S6 = "S6"
    S7 = "S7"
    S8 = "S8"


TransitionMatrix = dict[MarkovState, dict[MarkovState, float]]
TERMINAL_STATES = (MarkovState.S7, MarkovState.S8)


@dataclass(frozen=True)
class WorkflowResult:
    terminal_probabilities: dict[MarkovState, float]
    unverified_action_probability: float


def _coerce_state(value: MarkovState | str) -> MarkovState:
    try:
        return value if isinstance(value, MarkovState) else MarkovState(value)
    except ValueError as exc:
        raise ValueError(f"unknown Markov state: {value!r}") from exc


def validate_transition_matrix(
    matrix: Mapping[MarkovState | str, Mapping[MarkovState | str, float]],
    *,
    tolerance: float = 1e-9,
) -> TransitionMatrix:
    normalized: TransitionMatrix = {}

    for raw_from_state, raw_edges in matrix.items():
        from_state = _coerce_state(raw_from_state)
        if not raw_edges:
            raise ValueError(f"transition row for {from_state.value} is empty")

        row: dict[MarkovState, float] = {}
        for raw_to_state, probability in raw_edges.items():
            to_state = _coerce_state(raw_to_state)
            probability = float(probability)
            if probability < 0:
                raise ValueError(
                    f"transition row for {from_state.value} contains a negative probability"
                )
            row[to_state] = probability

        row_total = sum(row.values())
        if abs(row_total - 1.0) > tolerance:
            raise ValueError(f"transition row for {from_state.value} must sum to 1")
        normalized[from_state] = row

    missing_rows = [state.value for state in MarkovState if state not in normalized]
    if missing_rows:
        raise ValueError(f"transition matrix missing rows: {', '.join(missing_rows)}")

    for terminal_state in TERMINAL_STATES:
        if normalized[terminal_state] != {terminal_state: 1.0}:
            raise ValueError(
                f"terminal state {terminal_state.value} must self-loop with probability 1"
            )

    return normalized


def _hitting_probability(
    matrix: TransitionMatrix,
    *,
    start_state: MarkovState,
    hit_state: MarkovState,
) -> float:
    absorbing = set(TERMINAL_STATES) | {hit_state}
    transient_states = [state for state in MarkovState if state not in absorbing]
    index = {state: position for position, state in enumerate(transient_states)}

    if start_state == hit_state:
        return 1.0
    if start_state in TERMINAL_STATES:
        return 0.0

    q = np.zeros((len(transient_states), len(transient_states)))
    r = np.zeros(len(transient_states))

    for from_state in transient_states:
        row = index[from_state]
        for to_state, probability in matrix[from_state].items():
            if to_state == hit_state:
                r[row] += probability
            elif to_state in index:
                q[row, index[to_state]] += probability

    solution = np.linalg.solve(np.eye(len(transient_states)) - q, r)
    return float(solution[index[start_state]])


def evaluate_markov_workflow(
    matrix: Mapping[MarkovState | str, Mapping[MarkovState | str, float]],
    *,
    start_state: MarkovState | str = MarkovState.S0,
) -> WorkflowResult:
    normalized = validate_transition_matrix(matrix)
    start = _coerce_state(start_state)
    transient_states = [state for state in MarkovState if state not in TERMINAL_STATES]
    terminal_states = list(TERMINAL_STATES)
    transient_index = {state: position for position, state in enumerate(transient_states)}
    terminal_index = {state: position for position, state in enumerate(terminal_states)}

    if start in terminal_index:
        terminal_probabilities = {state: 0.0 for state in terminal_states}
        terminal_probabilities[start] = 1.0
        return WorkflowResult(
            terminal_probabilities=terminal_probabilities,
            unverified_action_probability=0.0,
        )

    q = np.zeros((len(transient_states), len(transient_states)))
    r = np.zeros((len(transient_states), len(terminal_states)))

    for from_state in transient_states:
        row = transient_index[from_state]
        for to_state, probability in normalized[from_state].items():
            if to_state in terminal_index:
                r[row, terminal_index[to_state]] += probability
            else:
                q[row, transient_index[to_state]] += probability

    absorption = np.linalg.solve(np.eye(len(transient_states)) - q, r)
    terminal_probabilities = {
        terminal_state: float(absorption[transient_index[start], terminal_index[terminal_state]])
        for terminal_state in terminal_states
    }

    return WorkflowResult(
        terminal_probabilities=terminal_probabilities,
        unverified_action_probability=_hitting_probability(
            normalized,
            start_state=start,
            hit_state=MarkovState.S2,
        ),
    )
