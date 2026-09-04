from __future__ import annotations

from datetime import UTC, datetime
from urllib.error import URLError

import numpy as np
import pandas as pd
import pytest

from cloud2watt.data.paired import (
    HISTORY_MINUTES,
    build_sample_index,
    datetime_index_to_ns,
    discover_paired_window,
    match_satellite_indices,
    paired_time_grid,
    partition_satellite_times,
    select_compact_sites,
    validate_sample_index,
)
from cloud2watt.data.seviri import SeviriStore, SitePixel, reshape_zarr_chunk


def test_time_grid_has_seven_days_of_issues_and_deduplicated_history() -> None:
    start = datetime(2020, 12, 1, tzinfo=UTC)
    end = datetime(2020, 12, 8, tzinfo=UTC)
    issues, frames = paired_time_grid(start, end)

    assert len(issues) == 7 * 24 * 4
    assert len(frames) == len(issues) + 3
    assert issues[0] == pd.Timestamp("2020-12-01 00:15", tz="UTC")
    assert issues[-1] == pd.Timestamp("2020-12-08 00:00", tz="UTC")
    assert frames[0] == issues[0] + pd.Timedelta(minutes=min(HISTORY_MINUTES))
    restored = pd.to_datetime(datetime_index_to_ns(frames), unit="ns", utc=True)
    assert pd.DatetimeIndex(restored).equals(frames.as_unit("ns"))


def test_satellite_matching_is_exact_and_never_uses_nearest_future() -> None:
    available = pd.date_range("2020-12-01", periods=3, freq="15min", tz="UTC")
    requested = pd.DatetimeIndex([available[0], available[2]])
    assert match_satellite_indices(requested, available) == [0, 2]
    with pytest.raises(ValueError, match="missing 1"):
        match_satellite_indices(
            pd.DatetimeIndex([pd.Timestamp("2020-12-01 00:14", tz="UTC")]), available
        )
    present, indices, missing = partition_satellite_times(
        pd.DatetimeIndex([available[0], pd.Timestamp("2020-12-01 00:14", tz="UTC")]),
        available,
    )
    assert present.tolist() == [available[0]]
    assert indices == [0]
    assert missing.tolist() == [pd.Timestamp("2020-12-01 00:14", tz="UTC")]


def test_padded_zarr_edge_chunk_is_trimmed_to_logical_shape() -> None:
    values = np.arange(12)
    spec = {"shape": [10], "chunks": [6], "order": "C"}
    result = reshape_zarr_chunk(values[6:], spec, [1])
    np.testing.assert_array_equal(result, np.array([6, 7, 8, 9]))


def test_seviri_crop_preserves_channel_y_x_order_across_chunks() -> None:
    store = object.__new__(SeviriStore)
    store.data_spec = {"shape": [1, 6, 6, 3], "chunks": [1, 4, 4, 3]}

    def decode_chunk(_array_name: str, chunk_key: str) -> np.ndarray:
        _, y_chunk, x_chunk, _ = map(int, chunk_key.split("."))
        values = np.empty((1, 4, 4, 3), dtype=np.float16)
        for y in range(4):
            for x in range(4):
                global_y = y_chunk * 4 + y
                global_x = x_chunk * 4 + x
                values[0, y, x] = [global_y * 10 + global_x + channel * 100 for channel in range(3)]
        return values

    store.decode_chunk = decode_chunk
    crop = store.read_crop(
        time_index=0,
        site=SitePixel("one", 3, 3, 0.0, 0.0),
        channel_indices=[2, 0],
        height=4,
        width=4,
    )
    assert crop.shape == (2, 4, 4)
    assert crop[0, 0, 0] == 211
    assert crop[1, -1, -1] == 44


def test_remote_chunk_fetch_retries_and_caches(monkeypatch, tmp_path) -> None:
    calls = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return b"chunk"

    def fake_urlopen(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise URLError("temporary")
        return Response()

    monkeypatch.setattr("cloud2watt.data.seviri.urlopen", fake_urlopen)
    monkeypatch.setattr("cloud2watt.data.seviri.time.sleep", lambda _seconds: None)
    store = object.__new__(SeviriStore)
    store.store_url = "https://example.invalid/store"
    store.cache_dir = tmp_path
    store.timeout = 1
    store.max_attempts = 3
    store.downloaded_bytes = 0
    store.cache_hits = 0
    store.network_fetches = 0
    store.retry_count = 0

    assert store._fetch_bytes("data/0.0.0.0") == b"chunk"
    assert store.retry_count == 1
    assert store.network_fetches == 1
    assert store._fetch_bytes("data/0.0.0.0") == b"chunk"
    assert store.cache_hits == 1
    assert calls == 2


def test_compact_site_selection_prefers_coverage_then_id() -> None:
    times = pd.date_range("2020-12-01 00:05", periods=12, freq="5min", tz="UTC")
    power = pd.DataFrame(
        [
            {"ss_id": site_id, "datetime_GMT": timestamp}
            for site_id in (3, 1, 2)
            for timestamp in times
        ]
    )
    metadata = pd.DataFrame(
        {
            "ss_id": [3, 1, 2],
            "latitude_rounded": [51.6, 51.61, 51.59],
            "longitude_rounded": [-1.8, -1.79, -1.81],
            "kWp": [2.0, 2.0, 2.0],
            "orientation": [180.0, 180.0, 180.0],
            "tilt": [30.0, 30.0, 30.0],
        }
    )
    selected = select_compact_sites(
        power,
        metadata,
        start_utc=datetime(2020, 12, 1, tzinfo=UTC),
        end_utc=datetime(2020, 12, 1, 1, tzinfo=UTC),
        site_count=2,
    )
    assert selected["ss_id"].tolist() == [1, 2]


def test_overlap_discovery_skips_incomplete_satellite_day() -> None:
    power_times = pd.date_range("2020-11-30 23:15", "2020-12-04", freq="5min", tz="UTC")
    power = pd.DataFrame(
        [
            {"ss_id": site_id, "datetime_GMT": timestamp}
            for site_id in (1, 2)
            for timestamp in power_times
        ]
    )
    metadata = pd.DataFrame(
        {
            "ss_id": [1, 2],
            "latitude_rounded": [51.6, 51.61],
            "longitude_rounded": [-1.8, -1.79],
            "kWp": [2.0, 2.0],
            "orientation": [180.0, 180.0],
            "tilt": [30.0, 30.0],
        }
    )
    satellite = pd.date_range("2020-12-01 23:30", "2020-12-04", freq="15min", tz="UTC")
    start, end, sites = discover_paired_window(power, metadata, satellite, days=1, site_count=2)
    assert start == datetime(2020, 12, 2, tzinfo=UTC)
    assert end == datetime(2020, 12, 3, tzinfo=UTC)
    assert len(sites) == 2


def test_sample_index_uses_only_valid_history_and_masks_targets() -> None:
    issue = pd.DatetimeIndex([pd.Timestamp("2020-12-01 01:00", tz="UTC")])
    frames = pd.date_range("2020-12-01 00:15", periods=4, freq="15min", tz="UTC")
    power_times = list(frames) + [
        issue[0] + pd.Timedelta(minutes=offset) for offset in (15, 30, 60, 120, 180, 240)
    ]
    power = pd.DataFrame(
        {
            "ss_id": [1] * len(power_times),
            "datetime_GMT": power_times,
            "normalized_power": np.arange(len(power_times), dtype=float),
            "is_valid": [True] * (len(power_times) - 1) + [False],
        }
    )
    sites = pd.DataFrame({"ss_id": [1]})
    index, stats = build_sample_index(power, sites, issue_times=issue, satellite_times=frames)
    assert len(index) == 1
    assert index.iloc[0]["satellite_frame_indices"] == [0, 1, 2, 3]
    assert index.iloc[0]["target_mask"] == [True, True, True, True, True, False]
    assert stats == {
        "dropped_invalid_history": 0,
        "dropped_missing_satellite_history": 0,
        "dropped_invalid_satellite_history": 0,
    }
    assert validate_sample_index(index, power, frames) == {
        "validated_samples": 1,
        "validated_satellite_references": 4,
        "validated_power_references": 10,
    }

    power.loc[0, "is_valid"] = False
    index, stats = build_sample_index(power, sites, issue_times=issue, satellite_times=frames)
    assert index.empty
    assert stats == {
        "dropped_invalid_history": 1,
        "dropped_missing_satellite_history": 0,
        "dropped_invalid_satellite_history": 0,
    }

    power.loc[0, "is_valid"] = True
    index, stats = build_sample_index(
        power,
        sites,
        issue_times=issue,
        satellite_times=frames,
        invalid_satellite_keys={("1", frames[0])},
    )
    assert index.empty
    assert stats["dropped_invalid_satellite_history"] == 1
