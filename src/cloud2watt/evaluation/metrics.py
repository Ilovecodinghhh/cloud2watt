"""Masked forecast and ramp-event metrics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _masked_values(
    observed: np.ndarray, predicted: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if observed.shape != predicted.shape or observed.shape != mask.shape:
        raise ValueError("observed, predicted, and mask must have the same shape")
    valid = mask & np.isfinite(observed) & np.isfinite(predicted)
    if not valid.any():
        raise ValueError("metric has no valid targets")
    return observed[valid], predicted[valid]


def masked_mae(observed: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> float:
    truth, forecast = _masked_values(observed, predicted, mask)
    return float(np.mean(np.abs(truth - forecast)))


def masked_rmse(observed: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> float:
    truth, forecast = _masked_values(observed, predicted, mask)
    return float(np.sqrt(np.mean(np.square(truth - forecast))))


def masked_nmae(observed: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> float:
    """MAE of capacity-normalized power, equivalent to capacity-normalized MAE."""
    return masked_mae(observed, predicted, mask)


def skill_score(model_mae: float, persistence_mae: float) -> float:
    if not np.isfinite(persistence_mae) or persistence_mae <= 0:
        raise ValueError("persistence MAE must be finite and positive")
    return float(1.0 - model_mae / persistence_mae)


def ramp_labels(current: np.ndarray, future_30m: np.ndarray, threshold: float = 0.20) -> np.ndarray:
    changes = np.asarray(future_30m, dtype=float) - np.asarray(current, dtype=float)
    return np.where(changes >= threshold, 1, np.where(changes <= -threshold, -1, 0))


def event_precision_recall_f1(
    times: Iterable[object],
    observed_events: np.ndarray,
    predicted_events: np.ndarray,
    *,
    direction: int,
    tolerance_minutes: int = 15,
) -> dict[str, float | int]:
    """Score one ramp direction with one-to-one temporal tolerance matching."""
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    timestamps = pd.to_datetime(list(times), utc=True)
    observed = list(timestamps[np.asarray(observed_events) == direction])
    predicted = list(timestamps[np.asarray(predicted_events) == direction])
    tolerance = pd.Timedelta(minutes=tolerance_minutes)
    unmatched = set(range(len(observed)))
    true_positive = 0
    for prediction in predicted:
        candidates = [
            index for index in unmatched if abs(observed[index] - prediction) <= tolerance
        ]
        if candidates:
            best = min(candidates, key=lambda index: abs(observed[index] - prediction))
            unmatched.remove(best)
            true_positive += 1
    false_positive = len(predicted) - true_positive
    false_negative = len(observed) - true_positive
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(observed) if observed else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
