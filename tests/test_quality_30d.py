from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from cloud2watt.data.alignment import aggregate_power_15min
from cloud2watt.data.quality import (
    QualityFlag,
    duplicate_timestamp_keys,
    flag_context_quality,
    flag_power_quality,
    missing_site_metadata,
    quality_breakdown,
)
from cloud2watt.data.solar import build_solar_features


def test_scalar_power_quality_rules_have_hit_and_non_hit() -> None:
    frame = pd.DataFrame({"normalized_power": [np.nan, -0.1, 1.6, 0.5]})
    result, stats = flag_power_quality(frame)
    assert stats.as_dict() == {"above_capacity_limit": 1, "missing": 1, "negative": 1}
    assert result["is_valid"].tolist() == [False, False, False, True]


def test_duplicate_raw_timestamp_is_collapsed_and_auditable() -> None:
    times = pd.to_datetime(
        ["2020-01-01 00:05Z", "2020-01-01 00:05Z", "2020-01-01 00:10Z", "2020-01-01 00:15Z"]
    )
    raw = pd.DataFrame({"ss_id": [1] * 4, "datetime_GMT": times, "generation_Wh": [1, 3, 2, 3]})
    assert len(duplicate_timestamp_keys(raw)) == 2
    aligned = aggregate_power_15min(raw, pd.Series({1: 1.0}), value_semantics="interval_energy")
    assert len(aligned) == 1
    assert bool(aligned.loc[0, "raw_duplicate"])


def test_night_nonzero_and_daytime_stuck_zero_rules() -> None:
    times = pd.date_range("2020-01-01", periods=11, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "ss_id": [1] * 11,
            "datetime_GMT": times,
            "normalized_power": [0.02, 0.0, 0.0] + [0.0] * 8,
            "solar_elevation_deg": [-5.0, -5.0, 15.0] + [20.0] * 8,
            "quality_flags": np.zeros(11, dtype="uint16"),
        }
    )
    result, stats = flag_context_quality(frame, stuck_zero_bins=8)
    assert stats.as_dict() == {"daytime_stuck_zero": 9, "night_nonzero": 1}
    assert result.loc[1, "is_valid"]
    assert result.loc[0, "quality_flags"] & int(QualityFlag.NIGHT_NONZERO)


def test_missing_site_metadata_rule() -> None:
    sites = pd.DataFrame(
        {
            "latitude_rounded": [52.0, np.nan],
            "longitude_rounded": [-1.0, -1.0],
            "kWp": [2.0, 0.0],
            "tilt": [30.0, 30.0],
            "orientation": [180.0, 180.0],
        }
    )
    assert missing_site_metadata(sites).tolist() == [False, True]


def test_quality_breakdown_reports_rule_site_and_date() -> None:
    frame = pd.DataFrame(
        {
            "ss_id": [1, 2],
            "datetime_GMT": pd.to_datetime(["2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"]),
            "quality_flags": [int(QualityFlag.NEGATIVE), 0],
        }
    )
    report = quality_breakdown(frame)
    assert report["by_rule"]["negative"] == 1
    assert report["by_site"]["1"]["negative"] == 1
    assert report["by_date"]["2020-01-01"]["negative"] == 1


def test_solar_features_cover_day_and_night_and_require_timezone() -> None:
    sites = pd.DataFrame({"ss_id": [1], "latitude_rounded": [51.5], "longitude_rounded": [0.0]})
    times = pd.DatetimeIndex(pd.to_datetime(["2020-06-21 00:00Z", "2020-06-21 12:00Z"]))
    features = build_solar_features(times, sites)
    assert features.loc[0, "solar_elevation_deg"] < 0
    assert features.loc[1, "solar_elevation_deg"] > 55
    assert features.loc[0, "clear_sky_ghi_wm2"] == pytest.approx(0.0)
    assert 0 <= features.loc[1, "solar_azimuth_deg"] <= 360
    with pytest.raises(ValueError, match="timezone-aware"):
        build_solar_features(pd.date_range("2020-01-01", periods=1), sites)


def test_thirty_day_grid_has_expected_scale() -> None:
    from cloud2watt.data.paired import paired_time_grid

    start = datetime(2020, 12, 1, tzinfo=UTC)
    issues, frames = paired_time_grid(start, start + pd.Timedelta(days=30))
    assert len(issues) * 20 == 57_600
    assert len(frames) == 2_883
