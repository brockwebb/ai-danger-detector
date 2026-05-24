from __future__ import annotations

import numpy as np


def _arrays(predicted, observed) -> tuple[np.ndarray, np.ndarray]:
    predicted_array = np.asarray(predicted, dtype=float)
    observed_array = np.asarray(observed, dtype=float)
    if predicted_array.shape != observed_array.shape:
        raise ValueError("predicted and observed values must have the same shape")
    if predicted_array.size == 0:
        raise ValueError("metric inputs must not be empty")
    return predicted_array, observed_array


def brier_score(probabilities, outcomes) -> float:
    probabilities, outcomes = _arrays(probabilities, outcomes)
    return float(np.mean((probabilities - outcomes) ** 2))


def log_loss(probabilities, outcomes, *, epsilon: float = 1e-15) -> float:
    probabilities, outcomes = _arrays(probabilities, outcomes)
    probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)
    losses = -(
        outcomes * np.log(probabilities)
        + (1.0 - outcomes) * np.log(1.0 - probabilities)
    )
    return float(np.mean(losses))


def expected_calibration_error(probabilities, outcomes, *, bins: int = 10) -> float:
    probabilities, outcomes = _arrays(probabilities, outcomes)
    if bins <= 0:
        raise ValueError("bins must be positive")

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = probabilities.size
    error = 0.0

    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not np.any(mask):
            continue
        confidence = float(np.mean(probabilities[mask]))
        accuracy = float(np.mean(outcomes[mask]))
        error += (np.sum(mask) / total) * abs(confidence - accuracy)

    return float(error)


def false_reassurance_rate(predicted_scores, true_labels, *, threshold: float) -> float:
    predicted_scores, true_labels = _arrays(predicted_scores, true_labels)
    positives = true_labels == 1
    if not np.any(positives):
        return 0.0
    return float(np.mean(predicted_scores[positives] < threshold))


def false_escalation_rate(predicted_scores, true_labels, *, threshold: float) -> float:
    predicted_scores, true_labels = _arrays(predicted_scores, true_labels)
    negatives = true_labels == 0
    if not np.any(negatives):
        return 0.0
    return float(np.mean(predicted_scores[negatives] >= threshold))


def interval_coverage(lower, upper, observed) -> float:
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    observed_array = np.asarray(observed, dtype=float)
    if lower_array.shape != upper_array.shape or lower_array.shape != observed_array.shape:
        raise ValueError("lower, upper, and observed values must have the same shape")
    if lower_array.size == 0:
        raise ValueError("metric inputs must not be empty")
    inside = (observed_array >= lower_array) & (observed_array <= upper_array)
    return float(np.mean(inside))
