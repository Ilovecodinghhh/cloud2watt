from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
import zarr

from cloud2watt.data.alignment import aggregate_power_15min
from cloud2watt.data.manifest import build_fixture_manifest
from cloud2watt.data.mini import build_power_mini_dataset, mark_source_bad_periods
from cloud2watt.data.quality import QualityFlag, flag_power_quality
from cloud2watt.data.schema import ForecastSample, SampleMetadata
from cloud2watt.data.spatial import center_crop
from cloud2watt.data.storage import write_derived_dataset


def test_aggregate_power_uses_right_closed_bins_and_normalizes_capacity() -> None:
    frame = pd.DataFrame(
        {
            "ss_id": [7, 7, 7, 7],
            "datetime_GMT": [
                "2020-12-01 00:05",
                "2020-12-01 00:10",
                "2020-12-01 00:15",
                "2020-12-01 00:20",
            ],
            "generation_Wh": [100.0, 100.0, 100.0, 100.0],
        }
    )
    result = aggregate_power_15min(frame, pd.Series({7: 2.0}))

    assert result.loc[0, "datetime_GMT"] == pd.Timestamp("2020-12-01 00:15", tz="UTC")
    assert result.loc[0, "normalized_power"] == pytest.approx(0.6)
    assert pd.isna(result.loc[1, "normalized_power"])


def test_quality_flags_are_combined_and_counted() -> None:
    frame = pd.DataFrame({"normalized_power": [np.nan, -0.1, 0.5, 1.6]})
    flagged, stats = flag_power_quality(frame)

    assert flagged["quality_flags"].tolist() == [
        int(QualityFlag.MISSING),
        int(QualityFlag.NEGATIVE),
        int(QualityFlag.OK),
        int(QualityFlag.ABOVE_CAPACITY_LIMIT),
    ]
    assert stats.as_dict() == {"above_capacity_limit": 1, "missing": 1, "negative": 1}


def test_source_bad_periods_are_inclusive_and_site_specific() -> None:
    power = pd.DataFrame(
        {
            "ss_id": [1, 1, 2],
            "datetime_GMT": ["2020-12-01 00:15", "2020-12-01 00:30", "2020-12-01 00:15"],
            "normalized_power": [0.1, 0.2, 0.1],
            "quality_flags": [0, 0, 0],
        }
    )
    bad = pd.DataFrame(
        {
            "ss_id": [1],
            "start_datetime_GMT": ["2020-12-01 00:15"],
            "end_datetime_GMT": ["2020-12-01 00:15"],
        }
    )
    result = mark_source_bad_periods(power, bad)
    assert result["quality_flags"].tolist() == [int(QualityFlag.SOURCE_BAD_PERIOD), 0, 0]


def test_center_crop_is_exact_and_rejects_edges() -> None:
    image = np.arange(7 * 9).reshape(7, 9)
    crop = center_crop(
        image,
        x_coordinates=np.arange(9),
        y_coordinates=np.arange(7),
        center_x=4.1,
        center_y=3.1,
        height=3,
        width=5,
    )
    np.testing.assert_array_equal(crop, image[2:5, 2:7])
    with pytest.raises(ValueError, match="beyond"):
        center_crop(
            image,
            x_coordinates=np.arange(9),
            y_coordinates=np.arange(7),
            center_x=0,
            center_y=0,
            height=3,
            width=3,
        )


def test_sample_schema_and_zarr_parquet_round_trip(tmp_path) -> None:
    sample = ForecastSample(
        satellite=np.zeros((4, 3, 8, 8), dtype="float32"),
        power_history=np.arange(4, dtype="float32"),
        solar_features=np.zeros((6, 2), dtype="float32"),
        site_features=np.array([2.5, 30, 180, 52, -1], dtype="float32"),
        target_power=np.array([0.4, 0.5], dtype="float32"),
        target_mask=np.array([True, True]),
        metadata=SampleMetadata("fixture-001", datetime(2020, 12, 1, tzinfo=UTC)),
    )
    output = tmp_path / "derived"
    write_derived_dataset(output, [sample], manifest={"schema_version": 1})

    root = zarr.open_group(output / "samples.zarr", mode="r")
    assert root["satellite"].shape == (1, 4, 3, 8, 8)
    index = pd.read_parquet(output / "sample_index.parquet")
    assert index.loc[0, "site_id"] == "fixture-001"
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == {
        "schema_version": 1
    }


def test_twenty_site_seven_day_manifest_contract() -> None:
    path = "tests/fixtures/mini_manifest.json"
    fixture = json.loads(open(path, encoding="utf-8").read())
    generated = build_fixture_manifest(
        [f"fixture-{index:03d}" for index in range(1, 21)],
        start_utc=datetime(2020, 12, 1, tzinfo=UTC),
    )
    assert generated == fixture


def test_end_to_end_power_mini_dataset() -> None:
    times = pd.date_range("2020-12-01 00:05", periods=12, freq="5min")
    power = pd.DataFrame(
        [
            {"ss_id": site_id, "datetime_GMT": time, "generation_Wh": 10.0}
            for site_id in (1, 2)
            for time in times
        ]
    )
    metadata = pd.DataFrame(
        {
            "ss_id": [1, 2],
            "latitude_rounded": [52.0, 53.0],
            "longitude_rounded": [-1.0, -2.0],
            "kWp": [1.0, 2.0],
        }
    )
    derived, sites, manifest = build_power_mini_dataset(
        power,
        metadata,
        start_utc=datetime(2020, 12, 1, tzinfo=UTC),
        days=1,
        site_count=2,
    )
    assert sites["ss_id"].tolist() == [1, 2]
    assert derived["datetime_GMT"].dt.tz is not None
    assert manifest["site_count"] == 2
    assert manifest["valid_rows"] == 8


def test_site_selection_ties_are_ordered_by_id() -> None:
    times = pd.date_range("2020-12-01 00:05", periods=3, freq="5min")
    power = pd.DataFrame(
        [
            {"ss_id": site_id, "datetime_GMT": time, "generation_Wh": 10.0}
            for site_id in (20, 10)
            for time in times
        ]
    )
    metadata = pd.DataFrame(
        {
            "ss_id": [20, 10],
            "latitude_rounded": [52.0, 53.0],
            "longitude_rounded": [-1.0, -2.0],
            "kWp": [1.0, 1.0],
        }
    )
    _, sites, _ = build_power_mini_dataset(
        power,
        metadata,
        start_utc=datetime(2020, 12, 1, tzinfo=UTC),
        days=1,
        site_count=2,
    )
    assert sites["ss_id"].tolist() == [10, 20]


def test_mini_dataset_uses_interval_end_window_boundaries() -> None:
    times = pd.date_range("2020-12-01", "2020-12-02", freq="5min")
    power = pd.DataFrame(
        {"ss_id": 1, "datetime_GMT": times, "generation_Wh": 10.0}
    )
    metadata = pd.DataFrame(
        {"ss_id": [1], "latitude_rounded": [52.0], "longitude_rounded": [-1.0], "kWp": [1.0]}
    )
    derived, _, _ = build_power_mini_dataset(
        power,
        metadata,
        start_utc=datetime(2020, 12, 1, tzinfo=UTC),
        days=1,
        site_count=1,
    )
    assert len(derived) == 96
    assert derived["datetime_GMT"].min() == pd.Timestamp("2020-12-01 00:15", tz="UTC")
    assert derived["datetime_GMT"].max() == pd.Timestamp("2020-12-02 00:00", tz="UTC")
