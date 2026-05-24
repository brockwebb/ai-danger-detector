from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Mapping

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


@dataclass(frozen=True)
class DistributionSpec:
    kind: Literal["fixed", "beta", "triangular", "lognormal"]
    parameters: dict[str, float]

    @classmethod
    def fixed(cls, value: float) -> DistributionSpec:
        return cls("fixed", {"value": float(value)})

    @classmethod
    def beta(cls, *, alpha: float, beta: float) -> DistributionSpec:
        return cls("beta", {"alpha": float(alpha), "beta": float(beta)})

    @classmethod
    def triangular(cls, *, left: float, mode: float, right: float) -> DistributionSpec:
        return cls(
            "triangular",
            {"left": float(left), "mode": float(mode), "right": float(right)},
        )

    @classmethod
    def lognormal(cls, *, mean: float, sigma: float) -> DistributionSpec:
        return cls("lognormal", {"mean": float(mean), "sigma": float(sigma)})

    def sample(self, rng: np.random.Generator, size: int | None = None) -> np.ndarray:
        if self.kind == "fixed":
            shape = () if size is None else size
            return np.full(shape, self.parameters["value"], dtype=float)
        if self.kind == "beta":
            return rng.beta(self.parameters["alpha"], self.parameters["beta"], size=size)
        if self.kind == "triangular":
            return rng.triangular(
                self.parameters["left"],
                self.parameters["mode"],
                self.parameters["right"],
                size=size,
            )
        if self.kind == "lognormal":
            return rng.lognormal(
                self.parameters["mean"], self.parameters["sigma"], size=size
            )
        raise ValueError(f"unknown distribution kind: {self.kind}")


@dataclass(frozen=True)
class SampledScenario:
    error_probability: float
    severity: float
    detectability: float
    reversibility: float
    verification_burden: float
    governance_strength: float
    conditional_loss: float


@dataclass(frozen=True)
class ScenarioSpec:
    error_probability: DistributionSpec
    severity: DistributionSpec
    detectability: DistributionSpec
    reversibility: DistributionSpec
    verification_burden: DistributionSpec
    governance_strength: DistributionSpec
    conditional_loss: DistributionSpec

    def sample(self, rng: np.random.Generator) -> SampledScenario:
        return SampledScenario(
            error_probability=float(self.error_probability.sample(rng)),
            severity=float(self.severity.sample(rng)),
            detectability=float(self.detectability.sample(rng)),
            reversibility=float(self.reversibility.sample(rng)),
            verification_burden=float(self.verification_burden.sample(rng)),
            governance_strength=float(self.governance_strength.sample(rng)),
            conditional_loss=float(self.conditional_loss.sample(rng)),
        )


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


def _clamp_probability(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _normalize_weights(weights: Mapping[MarkovState, float]) -> dict[MarkovState, float]:
    cleaned = {state: max(float(weight), 0.0) for state, weight in weights.items()}
    total = sum(cleaned.values())
    if total <= 0:
        share = 1.0 / len(cleaned)
        return {state: share for state in cleaned}
    return {state: weight / total for state, weight in cleaned.items()}


def build_transition_matrix(sampled: SampledScenario) -> TransitionMatrix:
    error_probability = _clamp_probability(sampled.error_probability)
    severity = _clamp_probability(sampled.severity)
    detectability = _clamp_probability(sampled.detectability)
    reversibility = _clamp_probability(sampled.reversibility)
    verification_burden = _clamp_probability(sampled.verification_burden)
    governance_strength = _clamp_probability(sampled.governance_strength)

    first_review = _normalize_weights(
        {
            MarkovState.S2: (1.0 - governance_strength) * (1.0 - 0.4 * detectability),
            MarkovState.S3: (0.25 + detectability) * (1.0 - 0.5 * governance_strength),
            MarkovState.S4: (0.15 + governance_strength)
            * (0.5 + 0.5 * verification_burden),
        }
    )

    checking_detection = _clamp_probability(
        0.15 + 0.65 * detectability - 0.25 * verification_burden
    )
    checking_escalation = _clamp_probability(
        0.10 + 0.45 * governance_strength + 0.20 * verification_burden
    )
    checking_action = max(0.0, 1.0 - checking_detection - checking_escalation)
    s3 = _normalize_weights(
        {
            MarkovState.S5: checking_detection,
            MarkovState.S4: checking_escalation,
            MarkovState.S6: checking_action,
        }
    )

    expert_detection = _clamp_probability(
        0.45 + 0.45 * detectability + 0.10 * governance_strength
    )
    s4 = {MarkovState.S5: expert_detection, MarkovState.S6: 1.0 - expert_detection}

    harm_after_action = _clamp_probability(
        error_probability * (0.25 + 0.75 * severity) * (1.0 - 0.55 * reversibility)
    )

    return validate_transition_matrix(
        {
            MarkovState.S0: {MarkovState.S1: 1.0},
            MarkovState.S1: first_review,
            MarkovState.S2: {MarkovState.S6: 1.0},
            MarkovState.S3: s3,
            MarkovState.S4: s4,
            MarkovState.S5: {MarkovState.S7: 0.85, MarkovState.S1: 0.15},
            MarkovState.S6: {
                MarkovState.S7: 1.0 - harm_after_action,
                MarkovState.S8: harm_after_action,
            },
            MarkovState.S7: {MarkovState.S7: 1.0},
            MarkovState.S8: {MarkovState.S8: 1.0},
        }
    )


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
