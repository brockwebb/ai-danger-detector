"""Bayesian calibration primitives for bounded ADD probabilities."""

from __future__ import annotations

from dataclasses import dataclass, field


def _as_tuple(values: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    return tuple(str(value).strip() for value in values if str(value).strip())


@dataclass(frozen=True)
class BetaPrior:
    """Traceable beta prior for one bounded probability parameter."""

    parameter_name: str
    alpha: float
    beta: float
    version: str = "unversioned"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not str(self.parameter_name).strip():
            raise ValueError("parameter_name must be non-empty")
        alpha = float(self.alpha)
        beta = float(self.beta)
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if beta <= 0:
            raise ValueError("beta must be positive")
        object.__setattr__(self, "parameter_name", str(self.parameter_name).strip())
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "version", str(self.version).strip() or "unversioned")
        object.__setattr__(self, "notes", _as_tuple(self.notes))


@dataclass(frozen=True)
class BetaObservation:
    """Weighted beta-binomial observation with source/evidence traceability."""

    successes: float
    failures: float
    weight: float = 1.0
    source_id: str | None = None
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        successes = float(self.successes)
        failures = float(self.failures)
        weight = float(self.weight)
        if successes < 0:
            raise ValueError("successes must be non-negative")
        if failures < 0:
            raise ValueError("failures must be non-negative")
        if weight < 0:
            raise ValueError("weight must be non-negative")
        source_id = None if self.source_id is None else str(self.source_id).strip()
        object.__setattr__(self, "successes", successes)
        object.__setattr__(self, "failures", failures)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "source_id", source_id or None)
        object.__setattr__(self, "evidence_ids", _as_tuple(self.evidence_ids))
        object.__setattr__(self, "notes", _as_tuple(self.notes))

    @property
    def weighted_successes(self) -> float:
        return self.successes * self.weight

    @property
    def weighted_failures(self) -> float:
        return self.failures * self.weight

    @property
    def effective_sample_size(self) -> float:
        return (self.successes + self.failures) * self.weight


@dataclass(frozen=True)
class BetaPosterior:
    """Posterior beta parameters plus the trace used to produce them."""

    alpha: float
    beta: float
    prior: BetaPrior
    observations: tuple[BetaObservation, ...]
    effective_sample_size: float
    source_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / ((total**2) * (total + 1.0))

    def summary(self) -> dict:
        return {
            "parameter_name": self.prior.parameter_name,
            "prior_version": self.prior.version,
            "alpha": self.alpha,
            "beta": self.beta,
            "mean": self.mean,
            "variance": self.variance,
            "effective_sample_size": self.effective_sample_size,
            "source_ids": self.source_ids,
            "evidence_ids": self.evidence_ids,
            "notes": self.notes,
        }


def update_beta_binomial(
    prior: BetaPrior,
    observations: tuple[BetaObservation, ...] | list[BetaObservation],
) -> BetaPosterior:
    observations = tuple(observations)
    if not observations:
        raise ValueError("at least one observation is required")
    alpha = prior.alpha + sum(
        observation.weighted_successes for observation in observations
    )
    beta = prior.beta + sum(observation.weighted_failures for observation in observations)
    effective_sample_size = sum(
        observation.effective_sample_size for observation in observations
    )
    source_ids = tuple(
        sorted(
            {
                observation.source_id
                for observation in observations
                if observation.source_id is not None
            }
        )
    )
    evidence_ids = tuple(
        sorted(
            {
                evidence_id
                for observation in observations
                for evidence_id in observation.evidence_ids
            }
        )
    )
    notes = tuple(note for observation in observations for note in observation.notes)
    return BetaPosterior(
        alpha=alpha,
        beta=beta,
        prior=prior,
        observations=observations,
        effective_sample_size=effective_sample_size,
        source_ids=source_ids,
        evidence_ids=evidence_ids,
        notes=notes,
    )
