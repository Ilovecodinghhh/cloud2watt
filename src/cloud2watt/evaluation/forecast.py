"""V2 scoring: fixed targets, daylight selection, and reproducible comparison keys."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from cloud2watt.data.paired import FORECAST_MINUTES

PROTOCOL = "c2w-eval-v2.1"
PRIMARY_HORIZONS = (30, 60, 120)
SELECTION_METRIC = "daylight_primary_mae"


class ForecastError(ValueError):
    """An auditable failure that must not shrink the scoring population."""

    def __init__(self, report: dict) -> None:
        self.report = report
        super().__init__(json.dumps(report, allow_nan=False))


def validate_forecast(prediction, target, mask, keys=None):
    prediction, target = np.asarray(prediction, float), np.asarray(target, float)
    mask = np.asarray(mask, bool)
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("prediction, target and mask shapes must match")
    if target.ndim != 2 or target.shape[1] != len(FORECAST_MINUTES):
        raise ValueError("forecast arrays must have shape [samples, 6]")
    bad_y, bad_p = mask & ~np.isfinite(target), mask & ~np.isfinite(prediction)
    if bad_y.any() or bad_p.any():
        failures = []
        for i, h in np.argwhere(bad_y | bad_p):
            item = {"row": int(i), "horizon_minutes": FORECAST_MINUTES[h]}
            if keys is not None:
                item |= {"site_id": str(keys.iloc[i].site_id),
                         "issue_time_utc": str(keys.iloc[i].issue_time_utc)}
            failures.append(item)
        raise ForecastError({"error": "non-finite valid targets/predictions",
                             "invalid_labels": int(bad_y.sum()),
                             "invalid_predictions": int(bad_p.sum()),
                             "valid_targets": int(mask.sum()),
                             "failure_rate": float((bad_y | bad_p).sum() / mask.sum()),
                             "keys": failures})
    return prediction, target, mask


def horizon_metrics(prediction, target, mask):
    prediction, target, mask = validate_forecast(prediction, target, mask)
    result = []
    for i, h in enumerate(FORECAST_MINUTES):
        error = prediction[mask[:, i], i] - target[mask[:, i], i]
        result.append({"horizon_minutes": h, "valid_targets": int(error.size),
                       "mae": float(np.abs(error).mean()) if error.size else None,
                       "nmae": float(np.abs(error).mean()) if error.size else None,
                       "rmse": float(np.sqrt(np.square(error).mean())) if error.size else None,
                       "bias": float(error.mean()) if error.size else None})
    return result


def comparison_identity(keys, target, mask, elevation):
    """Order-independent identity; masked-out target bytes have no effect."""
    frame = keys[["site_id", "issue_time_utc"]].copy()
    frame["site_id"] = frame.site_id.astype(str)
    frame["issue_time_utc"] = pd.to_datetime(frame.issue_time_utc, utc=True).astype(str)
    if frame.duplicated().any():
        raise ValueError("duplicate forecast sample keys")
    # Reset position indices independently of the caller's dataframe index.
    frame = frame.reset_index(drop=True)
    order = frame.sort_values(["site_id", "issue_time_utc"]).index.to_numpy()
    values = {"keys": frame.iloc[order].to_numpy().tolist(),
              "mask": mask[order].astype(int).tolist(),
              "targets": np.where(mask, target, 0)[order].tolist(),
              "daylight_mask": (mask & (elevation > 5))[order].astype(int).tolist()}
    return {f"{name}_sha256": hashlib.sha256(
        json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest() for name, value in values.items()}


def score_forecast(prediction, target, mask, elevation, keys, *, groups=True):
    prediction, target, mask = validate_forecast(prediction, target, mask, keys)
    elevation = np.asarray(elevation, float)
    if elevation.shape != mask.shape or np.any(mask & ~np.isfinite(elevation)):
        raise ValueError("valid targets require finite raw solar elevation")
    if len(keys) != len(target):
        raise ValueError("sample key count differs from predictions")
    daylight = mask & (elevation > 5.0)
    day_rows = horizon_metrics(prediction, target, daylight)
    selected = [row["mae"] for row in day_rows if row["horizon_minutes"] in PRIMARY_HORIZONS]
    primary = float(np.mean(selected)) if all(x is not None for x in selected) else None
    result = {"schema_version": 2, "protocol": PROTOCOL,
              "selection_metric": SELECTION_METRIC, SELECTION_METRIC: primary,
              "candidate_eligible": primary is not None,
              "valid_targets": int(mask.sum()), "prediction_failure_rate": 0.0,
              "all_day": horizon_metrics(prediction, target, mask), "daylight": day_rows,
              "comparison": comparison_identity(keys, target, mask, elevation),
              "solar_support": {
                  "night": (mask & (elevation <= 0)).sum(axis=0).tolist(),
                  "low_sun": (mask & (elevation > 0) & (elevation <= 5)).sum(axis=0).tolist(),
                  "daylight": daylight.sum(axis=0).tolist()},
              "sample_count": len(target),
              "date_count": int(pd.to_datetime(keys.issue_time_utc, utc=True).dt.date.nunique())}
    if groups:
        result["by_site"], result["by_issue_date"] = {}, {}
        labels = {"by_site": keys.site_id.astype(str).to_numpy(),
                  "by_issue_date": pd.to_datetime(keys.issue_time_utc, utc=True)
                  .dt.strftime("%Y-%m-%d").to_numpy()}
        for field, values in labels.items():
            for value in sorted(set(values)):
                subset = values == value
                report = score_forecast(prediction[subset], target[subset], mask[subset],
                                        elevation[subset], keys.loc[subset], groups=False)
                result[field][str(value)] = {"all_day": report["all_day"],
                                            "daylight": report["daylight"],
                                            SELECTION_METRIC: report[SELECTION_METRIC]}
        macro = []
        for i in range(6):
            values = [v["daylight"][i]["mae"] for v in result["by_site"].values()
                      if v["daylight"][i]["mae"] is not None]
            macro.append(float(np.mean(values)) if values else None)
        result["macro_site_daylight_mae"] = macro
        result["region_grouping"] = "unavailable: no frozen geographic groups in V1 dataset"
    return result


def require_primary(report):
    value = report[SELECTION_METRIC]
    if value is None or not np.isfinite(value):
        raise ValueError("no valid daylight targets in one or more primary horizons")
    return float(value)
