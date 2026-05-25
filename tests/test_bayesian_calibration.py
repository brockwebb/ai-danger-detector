import pytest

from core_model.bayesian_calibration import (
    BetaObservation,
    BetaPosterior,
    BetaPrior,
    update_beta_binomial,
)


def test_beta_prior_rejects_non_positive_parameters():
    with pytest.raises(ValueError, match="alpha must be positive"):
        BetaPrior(parameter_name="p_error_per_task", alpha=0, beta=2)

    with pytest.raises(ValueError, match="beta must be positive"):
        BetaPrior(parameter_name="p_error_per_task", alpha=2, beta=0)


def test_beta_observation_rejects_negative_counts_and_weights():
    with pytest.raises(ValueError, match="successes must be non-negative"):
        BetaObservation(successes=-1, failures=1)

    with pytest.raises(ValueError, match="failures must be non-negative"):
        BetaObservation(successes=1, failures=-1)

    with pytest.raises(ValueError, match="weight must be non-negative"):
        BetaObservation(successes=1, failures=1, weight=-0.5)


def test_update_beta_binomial_uses_unweighted_counts():
    prior = BetaPrior(
        parameter_name="p_error_per_task",
        alpha=2,
        beta=3,
        version="prior-v1",
    )
    observations = (
        BetaObservation(successes=3, failures=7, source_id="src-a"),
        BetaObservation(successes=1, failures=4, source_id="src-b"),
    )

    posterior = update_beta_binomial(prior, observations)

    assert isinstance(posterior, BetaPosterior)
    assert posterior.alpha == 6
    assert posterior.beta == 14
    assert posterior.effective_sample_size == 15
    assert posterior.mean == pytest.approx(0.3)
    assert posterior.variance == pytest.approx((6 * 14) / ((20**2) * 21))


def test_update_beta_binomial_applies_observation_weights():
    prior = BetaPrior(parameter_name="detectability", alpha=1, beta=1)
    observations = (
        BetaObservation(
            successes=8,
            failures=2,
            weight=0.25,
            source_id="src-weak",
            evidence_ids=("case-002", "case-001"),
            notes=("weak synthetic evidence",),
        ),
        BetaObservation(
            successes=4,
            failures=1,
            weight=2.0,
            source_id="src-strong",
            evidence_ids=("case-003",),
            notes=("adjudicated evidence",),
        ),
    )

    posterior = update_beta_binomial(prior, observations)

    assert posterior.alpha == pytest.approx(11.0)
    assert posterior.beta == pytest.approx(3.5)
    assert posterior.effective_sample_size == pytest.approx(12.5)
    assert posterior.source_ids == ("src-strong", "src-weak")
    assert posterior.evidence_ids == ("case-001", "case-002", "case-003")
    assert posterior.notes == ("weak synthetic evidence", "adjudicated evidence")


def test_posterior_summary_preserves_traceability_and_uncertainty():
    prior = BetaPrior(
        parameter_name="reversibility",
        alpha=4,
        beta=6,
        version="rev-prior-v1",
        notes=("expert elicited",),
    )
    observations = (
        BetaObservation(
            successes=2,
            failures=3,
            source_id="src-a",
            evidence_ids=("case-a",),
            notes=("retrospective review",),
        ),
    )

    posterior = update_beta_binomial(prior, observations)
    summary = posterior.summary()

    assert summary == {
        "parameter_name": "reversibility",
        "prior_version": "rev-prior-v1",
        "alpha": 6.0,
        "beta": 9.0,
        "mean": pytest.approx(0.4),
        "variance": pytest.approx((6 * 9) / ((15**2) * 16)),
        "effective_sample_size": 5.0,
        "source_ids": ("src-a",),
        "evidence_ids": ("case-a",),
        "notes": ("retrospective review",),
    }


def test_update_beta_binomial_rejects_empty_observations():
    prior = BetaPrior(parameter_name="p_error_per_task", alpha=1, beta=1)

    with pytest.raises(ValueError, match="at least one observation is required"):
        update_beta_binomial(prior, ())


def test_bayesian_calibration_exports_public_api():
    import core_model

    assert "BetaPrior" in core_model.__all__
    assert "BetaObservation" in core_model.__all__
    assert "BetaPosterior" in core_model.__all__
    assert "update_beta_binomial" in core_model.__all__
    assert core_model.BetaPrior is BetaPrior
    assert core_model.update_beta_binomial is update_beta_binomial
