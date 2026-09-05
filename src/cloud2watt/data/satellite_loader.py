"""Lazy, worker-safe loading for indexed satellite histories."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset, Sampler

from cloud2watt.data.paired import HISTORY_MINUTES
from cloud2watt.training import FeatureStatistics, PowerForecastDataset


@dataclass(frozen=True)
class SatelliteStatistics:
    """Training-only per-channel normalization statistics."""

    mean: list[float]
    scale: list[float]
    valid_pixels: list[int]
    frame_keys_sha256: str

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


def _training_frame_keys(samples: pd.DataFrame) -> list[tuple[int, int]]:
    keys = {
        (int(row.site_index), int(frame_index))
        for row in samples.itertuples(index=False)
        for frame_index in row.satellite_frame_indices
    }
    return sorted(keys)


def fit_satellite_statistics(
    samples: pd.DataFrame,
    zarr_path: Path,
    *,
    maximum_frames: int | None = None,
) -> SatelliteStatistics:
    """Stream unique training frames and fit finite-pixel channel statistics."""
    keys = _training_frame_keys(samples)
    if not keys:
        raise ValueError("satellite statistics require at least one training frame")
    if maximum_frames is not None:
        if maximum_frames < 1:
            raise ValueError("maximum_frames must be positive")
        positions = np.linspace(0, len(keys) - 1, min(maximum_frames, len(keys)), dtype=int)
        keys = [keys[position] for position in positions]
    frames = zarr.open_group(zarr_path, mode="r")["frames"]
    channel_count = int(frames.shape[2])
    sums = np.zeros(channel_count, dtype=np.float64)
    squares = np.zeros(channel_count, dtype=np.float64)
    counts = np.zeros(channel_count, dtype=np.int64)
    for site_index, frame_index in keys:
        frame = np.asarray(frames[site_index, frame_index], dtype=np.float32)
        for channel in range(channel_count):
            values = frame[channel]
            valid = np.isfinite(values)
            selected = values[valid].astype(np.float64)
            sums[channel] += selected.sum()
            squares[channel] += np.square(selected).sum()
            counts[channel] += selected.size
    if np.any(counts == 0):
        raise ValueError("one or more satellite channels contain no finite training pixels")
    mean = sums / counts
    variance = np.maximum(squares / counts - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale = np.where(scale > 1e-8, scale, 1.0)
    key_payload = json.dumps(keys, separators=(",", ":")).encode()
    return SatelliteStatistics(
        mean.tolist(), scale.tolist(), counts.tolist(), hashlib.sha256(key_payload).hexdigest()
    )


def consistent_spatial_jitter(
    history: np.ndarray, *, offset_y: int, offset_x: int, padding: int
) -> np.ndarray:
    """Apply one translation to every time and channel without changing shape."""
    if history.ndim != 4:
        raise ValueError("satellite history must have [time, channel, height, width] shape")
    if padding < 0 or abs(offset_y) > padding or abs(offset_x) > padding:
        raise ValueError("jitter offsets must lie inside padding")
    if padding == 0:
        return history.copy()
    padded = np.pad(history, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="edge")
    y_start = padding + offset_y
    x_start = padding + offset_x
    return padded[:, :, y_start:y_start + history.shape[2], x_start:x_start + history.shape[3]]


class SatelliteForecastDataset(Dataset):
    """Assemble indexed satellite and power fields without loading all frames into RAM."""

    def __init__(
        self,
        samples: pd.DataFrame,
        power: pd.DataFrame,
        sites: pd.DataFrame,
        feature_statistics: FeatureStatistics,
        satellite_statistics: SatelliteStatistics,
        zarr_path: Path,
        *,
        augment: bool = False,
        jitter_padding: int = 0,
        seed: int = 42,
    ) -> None:
        self.samples = samples.reset_index(drop=True).copy()
        self.power_dataset = PowerForecastDataset(
            self.samples, power, sites, feature_statistics
        )
        self.zarr_path = Path(zarr_path)
        self.statistics = satellite_statistics
        self.augment = augment
        self.jitter_padding = jitter_padding
        self.seed = seed
        self._root: Any | None = None
        self._validate_times()

    def _validate_times(self) -> None:
        root = zarr.open_group(self.zarr_path, mode="r")
        times = np.asarray(root["time_ns"][:], dtype=np.int64)
        for row in self.samples.itertuples(index=False):
            frame_indices = np.asarray(row.satellite_frame_indices, dtype=int)
            actual = pd.to_datetime(times[frame_indices], utc=True)
            expected = pd.DatetimeIndex(
                [pd.Timestamp(row.issue_time_utc) + pd.Timedelta(minutes=value)
                 for value in HISTORY_MINUTES]
            )
            actual_ns = actual.to_numpy(dtype="datetime64[ns]").astype("int64")
            expected_ns = expected.to_numpy(dtype="datetime64[ns]").astype("int64")
            if not np.array_equal(actual_ns, expected_ns) or actual.max() > pd.Timestamp(
                row.issue_time_utc
            ):
                raise ValueError("satellite history is misaligned or contains a future frame")

    def _frames(self) -> Any:
        if self._root is None:
            self._root = zarr.open_group(self.zarr_path, mode="r")
        return self._root["frames"]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_root"] = None
        return state

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples.iloc[index]
        frame_indices = np.asarray(row["satellite_frame_indices"], dtype=int)
        history = np.asarray(
            self._frames()[int(row["site_index"]), frame_indices], dtype=np.float32
        )
        mask = np.isfinite(history).reshape(history.shape[0], -1).all(axis=1)
        mean = np.asarray(self.statistics.mean, dtype=np.float32)[None, :, None, None]
        scale = np.asarray(self.statistics.scale, dtype=np.float32)[None, :, None, None]
        history = np.nan_to_num((history - mean) / scale)
        if self.augment and self.jitter_padding:
            rng = np.random.default_rng(self.seed + index)
            offset = rng.integers(-self.jitter_padding, self.jitter_padding + 1, size=2)
            history = consistent_spatial_jitter(
                history, offset_y=int(offset[0]), offset_x=int(offset[1]),
                padding=self.jitter_padding,
            )
        result = self.power_dataset[index]
        result["satellite"] = torch.from_numpy(np.ascontiguousarray(history))
        result["satellite_mask"] = torch.from_numpy(mask)
        result["satellite_frame_indices"] = torch.as_tensor(
            row["satellite_frame_indices"], dtype=torch.int64
        )
        return result


class ChunkBucketSampler(Sampler[int]):
    """Visit samples grouped by site and first Zarr time chunk."""

    def __init__(self, samples: pd.DataFrame, *, time_chunk: int = 12,
                 shuffle_buckets: bool = False, seed: int = 42) -> None:
        if time_chunk < 1:
            raise ValueError("time_chunk must be positive")
        frame = samples.reset_index(drop=True)
        buckets: dict[tuple[int, int], list[int]] = {}
        for index, row in enumerate(frame.itertuples(index=False)):
            key = (int(row.site_index), int(row.satellite_frame_indices[0]) // time_chunk)
            buckets.setdefault(key, []).append(index)
        keys = sorted(buckets)
        if shuffle_buckets:
            np.random.default_rng(seed).shuffle(keys)
        self.indices = [index for key in keys for index in buckets[key]]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)
