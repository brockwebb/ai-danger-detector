import pytest

from core_model.performance_metrics import (
    brier_score,
    expected_calibration_error,
    false_escalation_rate,
    false_reassurance_rate,
    interval_coverage,
    log_loss,
)


def test_brier_score_returns_mean_squared_probability_error():
    assert brier_score([0.1, 0.8, 0.6], [0, 1, 1]) == pytest.approx(0.07)


def test_log_loss_clips_extreme_probabilities():
    value = log_loss([0.0, 1.0], [0, 1])

    assert value < 1e-6


def test_expected_calibration_error_bins_predictions():
    value = expected_calibration_error(
        probabilities=[0.1, 0.2, 0.8, 0.9],
        outcomes=[0, 0, 1, 1],
        bins=2,
    )

    assert value == pytest.approx(0.15)


def test_false_reassurance_rate_counts_concern_cases_scored_too_low():
    rate = false_reassurance_rate(
        predicted_scores=[0.2, 0.4, 0.9, 0.1],
        true_labels=[1, 1, 1, 0],
        threshold=0.5,
    )

    assert rate == pytest.approx(2 / 3)


def test_false_escalation_rate_counts_low_concern_cases_scored_too_high():
    rate = false_escalation_rate(
        predicted_scores=[0.2, 0.7, 0.8, 0.1],
        true_labels=[0, 0, 1, 0],
        threshold=0.5,
    )

    assert rate == pytest.approx(1 / 3)


def test_interval_coverage_counts_outcomes_inside_interval():
    coverage = interval_coverage(
        lower=[0.1, 0.2, 0.4],
        upper=[0.5, 0.7, 0.9],
        observed=[0.3, 0.8, 0.9],
    )

    assert coverage == pytest.approx(2 / 3)
