"""Typed contracts for model-ready Cloud2Watt samples."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SampleMetadata:
    """Provenance and quality information for one forecast issue time."""

    site_id: str
    issue_time_utc: datetime
    source_versions: dict[str, str] = field(default_factory=dict)
    quality_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.issue_time_utc.tzinfo is None or self.issue_time_utc.utcoffset() is None:
            raise ValueError("issue_time_utc must be timezone-aware")
        if self.issue_time_utc.utcoffset().total_seconds() != 0:
            raise ValueError("issue_time_utc must be expressed in UTC")


@dataclass(frozen=True)
class ForecastSample:
    """A validated multimodal sample consumed by forecasting models."""

    satellite: np.ndarray
    power_history: np.ndarray
    solar_features: np.ndarray
    site_features: np.ndarray
    target_power: np.ndarray
    target_mask: np.ndarray
    metadata: SampleMetadata

    def __post_init__(self) -> None:
        if self.satellite.ndim != 4:
            raise ValueError("satellite must have shape [history, channels, height, width]")
        if self.power_history.ndim != 1:
            raise ValueError("power_history must be one-dimensional")
        if self.target_power.ndim != 1 or self.target_mask.ndim != 1:
            raise ValueError("target_power and target_mask must be one-dimensional")
        if self.target_power.shape != self.target_mask.shape:
            raise ValueError("target_power and target_mask shapes must match")
        if self.satellite.shape[0] != self.power_history.shape[0]:
            raise ValueError("satellite and power history must have equal history steps")
        expected_solar_steps = self.power_history.size + self.target_power.size
        if self.solar_features.ndim != 2 or self.solar_features.shape[0] != expected_solar_steps:
            raise ValueError("solar_features must cover history and forecast steps")
        if self.site_features.ndim != 1:
            raise ValueError("site_features must be one-dimensional")

    def index_record(self) -> dict[str, Any]:
        """Return scalar fields stored in the sample-index Parquet file."""
        return {
            "site_id": self.metadata.site_id,
            "issue_time_utc": self.metadata.issue_time_utc,
            "source_versions": json.dumps(
                self.metadata.source_versions, sort_keys=True, separators=(",", ":")
            ),
            "quality_flags": ",".join(self.metadata.quality_flags),
        }
