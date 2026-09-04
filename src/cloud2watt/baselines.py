"""Reference forecasting baselines."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd


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


def smart_persistence(
    current_power: float,
    current_clear_sky_ghi: float,
    future_clear_sky_ghi: Sequence[float],
    *,
    minimum_ghi: float = 20.0,
    upper_fraction: float = 1.5,
) -> npt.NDArray[np.float64]:
    """Hold the current clear-sky index constant over forecast horizons."""
    future = np.asarray(future_clear_sky_ghi, dtype=np.float64)
    if not np.isfinite(current_power) or not np.isfinite(future).all():
        raise ValueError("power and clear-sky irradiance must be finite")
    if current_clear_sky_ghi < minimum_ghi:
        return np.zeros_like(future)
    clear_sky_index = max(0.0, current_power / current_clear_sky_ghi)
    return np.clip(clear_sky_index * future, 0.0, upper_fraction)


class SiteTimeClimatology:
    """Training-only mean normalized power by site and 15-minute time slot."""

    def fit(self, power: pd.DataFrame) -> "SiteTimeClimatology":
        required = {"ss_id", "datetime_GMT", "normalized_power", "is_valid"}
        missing = required.difference(power.columns)
        if missing:
            raise ValueError(f"missing climatology columns: {sorted(missing)}")
        valid = power.loc[power["is_valid"]].copy()
        if valid.empty:
            raise ValueError("climatology training data has no valid rows")
        times = pd.to_datetime(valid["datetime_GMT"], utc=True)
        valid["slot"] = times.dt.hour * 4 + times.dt.minute // 15
        valid["site_id"] = valid["ss_id"].astype(str)
        self.by_site_slot = valid.groupby(["site_id", "slot"])["normalized_power"].mean()
        self.by_site = valid.groupby("site_id")["normalized_power"].mean()
        self.global_mean = float(valid["normalized_power"].mean())
        self.training_end_utc = times.max()
        return self

    def predict(self, site_id: str, times: Sequence[object]) -> npt.NDArray[np.float64]:
        if not hasattr(self, "by_site_slot"):
            raise ValueError("climatology must be fitted before prediction")
        timestamps = pd.to_datetime(list(times), utc=True)
        result = []
        site_fallback = float(self.by_site.get(str(site_id), self.global_mean))
        for timestamp in timestamps:
            slot = timestamp.hour * 4 + timestamp.minute // 15
            result.append(float(self.by_site_slot.get((str(site_id), slot), site_fallback)))
        return np.asarray(result, dtype=np.float64)
