from __future__ import annotations

from enum import Enum
from typing import Mapping


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
