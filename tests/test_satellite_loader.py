import pickle

import numpy as np
import pandas as pd
import pytest
import zarr
from torch.utils.data import DataLoader

from cloud2watt.data.satellite_loader import (
    ChunkBucketSampler,
    SatelliteForecastDataset,
    consistent_spatial_jitter,
    fit_satellite_statistics,
)
from cloud2watt.training import FeatureStatistics


def _fixture(tmp_path):
    times = pd.date_range("2021-01-01", periods=12, freq="15min", tz="UTC")
    root = zarr.open_group(tmp_path / "satellite.zarr", mode="w")
    frames = np.arange(1 * 12 * 3 * 8 * 8, dtype=np.float32).reshape(1, 12, 3, 8, 8)
    root.create_array("frames", data=frames, chunks=(1, 4, 3, 8, 8))
    root.create_array("time_ns", data=times.to_numpy(dtype="datetime64[ns]").astype("int64"))
    power = pd.DataFrame({
        "normalized_power": np.linspace(0.1, 0.8, 12),
        "solar_elevation_deg": np.linspace(1, 30, 12),
        "solar_azimuth_deg": np.linspace(100, 200, 12),
        "clear_sky_ghi_wm2": np.linspace(10, 500, 12),
    })
    sites = pd.DataFrame({
        "ss_id": ["a"], "kWp": [4.0], "tilt": [30.0], "orientation": [180.0],
        "latitude_rounded": [51.0], "longitude_rounded": [-1.0],
    })
    samples = pd.DataFrame({
        "site_id": ["a", "a"], "site_index": [0, 0],
        "issue_time_utc": [times[3], times[4]],
        "satellite_frame_indices": [np.arange(4), np.arange(1, 5)],
        "power_history_row_indices": [np.arange(4), np.arange(1, 5)],
        "target_row_indices": [np.arange(4, 10), np.arange(5, 11)],
        "target_mask": [np.ones(6, dtype=bool), np.ones(6, dtype=bool)],
    })
    return tmp_path / "satellite.zarr", samples, power, sites


def test_consistent_jitter_uses_same_translation_for_all_frames() -> None:
    history = np.stack([np.arange(16).reshape(1, 4, 4)] * 4)
    shifted = consistent_spatial_jitter(history, offset_y=1, offset_x=-1, padding=1)
    assert shifted.shape == history.shape
    np.testing.assert_array_equal(shifted[0], shifted[3])


def test_statistics_only_use_supplied_training_frames(tmp_path) -> None:
    path, samples, _, _ = _fixture(tmp_path)
    first = fit_satellite_statistics(samples.iloc[[0]], path)
    root = zarr.open_group(path, mode="a")
    root["frames"][0, 8:] = 1_000_000
    second = fit_satellite_statistics(samples.iloc[[0]], path)
    assert first == second


def test_lazy_dataset_schema_and_worker_pickle(tmp_path) -> None:
    path, samples, power, sites = _fixture(tmp_path)
    feature_stats = FeatureStatistics.fit(samples, power, sites)
    satellite_stats = fit_satellite_statistics(samples, path)
    dataset = SatelliteForecastDataset(
        samples, power, sites, feature_stats, satellite_stats, path
    )
    item = dataset[0]
    assert item["satellite"].shape == (4, 3, 8, 8)
    assert item["satellite_mask"].tolist() == [True] * 4
    assert item["satellite_frame_indices"].tolist() == [0, 1, 2, 3]
    assert pickle.loads(pickle.dumps(dataset))._root is None


def test_multiworker_loader_has_no_duplicate_samples(tmp_path) -> None:
    path, samples, power, sites = _fixture(tmp_path)
    dataset = SatelliteForecastDataset(
        samples, power, sites, FeatureStatistics.fit(samples, power, sites),
        fit_satellite_statistics(samples, path), path,
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=2)
    issue_times = [value for batch in loader for value in batch["issue_time_utc"]]
    assert issue_times == [value.isoformat() for value in samples["issue_time_utc"]]


def test_chunk_sampler_groups_site_and_time_chunk() -> None:
    samples = pd.DataFrame({
        "site_index": [1, 0, 0],
        "satellite_frame_indices": [np.array([13]), np.array([14]), np.array([1])],
    })
    assert list(ChunkBucketSampler(samples, time_chunk=12)) == [2, 1, 0]


def test_dataset_rejects_future_satellite_frame(tmp_path) -> None:
    path, samples, power, sites = _fixture(tmp_path)
    samples.at[0, "satellite_frame_indices"] = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError, match="future frame"):
        SatelliteForecastDataset(
            samples, power, sites, FeatureStatistics.fit(samples, power, sites),
            fit_satellite_statistics(samples, path), path,
        )
