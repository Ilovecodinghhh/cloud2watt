"""Reference forecasting baselines."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def persistence(last_observation: float, horizons: int) -> npt.NDArray[np.float64]:
    """Repeat the latest available power observation over every forecast horizon."""
    if horizons < 1:
        raise ValueError("horizons must be at least 1")
    if not np.isfinite(last_observation):
        raise ValueError("last_observation must be finite")
    return np.full(horizons, last_observation, dtype=np.float64)


def normalized_mae(observed: Sequence[float], predicted: Sequence[float], capacity: float) -> float:
    """Return mean absolute error normalized by installed capacity."""
    if capacity <= 0 or not np.isfinite(capacity):
        raise ValueError("capacity must be finite and positive")
    y_true = np.asarray(observed, dtype=np.float64)
    y_pred = np.asarray(predicted, dtype=np.float64)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("observed and predicted must have the same non-empty shape")
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("observed and predicted must contain only finite values")
    return float(np.mean(np.abs(y_true - y_pred)) / capacity)
