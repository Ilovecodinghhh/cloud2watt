"""Explicit quality flags and auditable filtering statistics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import IntFlag, auto

import pandas as pd


class QualityFlag(IntFlag):
    """Bit flags retained with derived power observations."""

    OK = 0
    MISSING = auto()
    NEGATIVE = auto()
    ABOVE_CAPACITY_LIMIT = auto()
    SOURCE_BAD_PERIOD = auto()


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
