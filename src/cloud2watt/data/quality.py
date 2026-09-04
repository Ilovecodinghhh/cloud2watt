"""Explicit quality flags and auditable filtering statistics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import IntFlag, auto

import numpy as np
import pandas as pd


class QualityFlag(IntFlag):
    """Bit flags retained with derived power observations."""

    OK = 0
    MISSING = auto()
    NEGATIVE = auto()
    ABOVE_CAPACITY_LIMIT = auto()
    SOURCE_BAD_PERIOD = auto()
    DUPLICATE_TIMESTAMP = auto()
    NIGHT_NONZERO = auto()
    DAYTIME_STUCK_ZERO = auto()
    SATELLITE_MISSING = auto()
    SATELLITE_INVALID = auto()
    SITE_METADATA_MISSING = auto()


@dataclass
class FilterStats:
    """Counts for every applied quality rule."""

    counts: Counter[str] = field(default_factory=Counter)

    def record(self, flag: QualityFlag, count: int) -> None:
        self.counts[flag.name.lower()] += int(count)

    def as_dict(self) -> dict[str, int]:
        return dict(sorted(self.counts.items()))


def flag_power_quality(
    frame: pd.DataFrame, *, upper_fraction: float = 1.5
) -> tuple[pd.DataFrame, FilterStats]:
    """Flag missing, negative, and implausibly high normalized observations."""
    if "normalized_power" not in frame:
        raise ValueError("normalized_power column is required")
    result = frame.copy()
    flags = pd.Series(0, index=result.index, dtype="uint16")
    stats = FilterStats()
    rules = (
        (QualityFlag.MISSING, result["normalized_power"].isna()),
        (QualityFlag.NEGATIVE, result["normalized_power"] < 0),
        (QualityFlag.ABOVE_CAPACITY_LIMIT, result["normalized_power"] > upper_fraction),
    )
    for flag, mask in rules:
        flags.loc[mask] = flags.loc[mask] | int(flag)
        stats.record(flag, int(mask.sum()))
    result["quality_flags"] = flags
    result["is_valid"] = flags == int(QualityFlag.OK)
    return result, stats


def duplicate_timestamp_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Return every duplicated raw site/timestamp key for audit reporting."""
    required = {"ss_id", "datetime_GMT"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing timestamp columns: {sorted(missing)}")
    normalized = frame.loc[:, ["ss_id", "datetime_GMT"]].copy()
    normalized["datetime_GMT"] = pd.to_datetime(normalized["datetime_GMT"], utc=True)
    return normalized.loc[normalized.duplicated(["ss_id", "datetime_GMT"], keep=False)].sort_values(
        ["ss_id", "datetime_GMT"], ignore_index=True
    )


def flag_context_quality(
    frame: pd.DataFrame,
    *,
    night_elevation_degrees: float = 0.0,
    night_nonzero_fraction: float = 0.01,
    daytime_elevation_degrees: float = 10.0,
    stuck_zero_bins: int = 8,
) -> tuple[pd.DataFrame, FilterStats]:
    """Flag physically suspicious PV observations using causal solar features."""
    required = {"ss_id", "datetime_GMT", "normalized_power", "solar_elevation_deg"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing context quality columns: {sorted(missing)}")
    result = frame.copy().sort_values(["ss_id", "datetime_GMT"], ignore_index=True)
    if "quality_flags" not in result:
        result["quality_flags"] = np.uint16(0)
    flags = result["quality_flags"].astype("uint16")
    stats = FilterStats()

    night = result["solar_elevation_deg"].le(night_elevation_degrees)
    night_nonzero = night & result["normalized_power"].gt(night_nonzero_fraction)
    flags.loc[night_nonzero] |= int(QualityFlag.NIGHT_NONZERO)
    stats.record(QualityFlag.NIGHT_NONZERO, int(night_nonzero.sum()))

    daytime_zero = result["solar_elevation_deg"].gt(daytime_elevation_degrees) & result[
        "normalized_power"
    ].fillna(np.nan).eq(0)
    stuck = pd.Series(False, index=result.index)
    for _, indices in result.groupby("ss_id", sort=False).groups.items():
        group_mask = daytime_zero.loc[indices]
        runs = group_mask.ne(group_mask.shift()).cumsum()
        lengths = group_mask.groupby(runs).transform("size")
        stuck.loc[indices] = group_mask & lengths.ge(stuck_zero_bins)
    flags.loc[stuck] |= int(QualityFlag.DAYTIME_STUCK_ZERO)
    stats.record(QualityFlag.DAYTIME_STUCK_ZERO, int(stuck.sum()))
    result["quality_flags"] = flags
    result["is_valid"] = flags.eq(int(QualityFlag.OK))
    return result, stats


def missing_site_metadata(sites: pd.DataFrame) -> pd.Series:
    """Identify sites missing coordinates, positive capacity, tilt, or orientation."""
    required = ["latitude_rounded", "longitude_rounded", "kWp", "tilt", "orientation"]
    absent = [column for column in required if column not in sites]
    if absent:
        return pd.Series(True, index=sites.index)
    numeric = sites[required].apply(pd.to_numeric, errors="coerce")
    return numeric.isna().any(axis=1) | numeric["kWp"].le(0)


def quality_breakdown(frame: pd.DataFrame) -> dict[str, object]:
    """Summarize bit flags by rule, site, and UTC date without double counting rows."""
    if frame.empty:
        return {"by_rule": {}, "by_site": {}, "by_date": {}}
    result: dict[str, object] = {"by_rule": {}, "by_site": {}, "by_date": {}}
    flags = frame["quality_flags"].astype("uint16")
    dates = pd.to_datetime(frame["datetime_GMT"], utc=True).dt.strftime("%Y-%m-%d")
    for flag in QualityFlag:
        if flag is QualityFlag.OK:
            continue
        mask = flags.map(lambda value, bit=int(flag): bool(value & bit))
        result["by_rule"][flag.name.lower()] = int(mask.sum())
        for site_id, count in frame.loc[mask].groupby("ss_id").size().items():
            result["by_site"].setdefault(str(site_id), {})[flag.name.lower()] = int(count)
        date_counts = frame.loc[mask].assign(_date=dates.loc[mask]).groupby("_date").size()
        for date, count in date_counts.items():
            result["by_date"].setdefault(str(date), {})[flag.name.lower()] = int(count)
    return result
