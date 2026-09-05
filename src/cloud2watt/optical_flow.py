"""Interpretable global cloud-motion features and ridge baseline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


def phase_correlation_translation(
    first: np.ndarray, second: np.ndarray, *, downsample: int = 4
) -> tuple[float, float]:
    """Estimate second-frame translation (dy, dx) using phase correlation."""
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("phase correlation requires equal two-dimensional frames")
    if downsample < 1:
        raise ValueError("downsample must be positive")
    a = np.nan_to_num(first[::downsample, ::downsample].astype(np.float32))
    b = np.nan_to_num(second[::downsample, ::downsample].astype(np.float32))
    a -= a.mean()
    b -= b.mean()
    cross = np.fft.fft2(b) * np.conj(np.fft.fft2(a))
    cross /= np.maximum(np.abs(cross), 1e-12)
    correlation = np.fft.ifft2(cross).real
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    if peak_y > correlation.shape[0] // 2:
        peak_y -= correlation.shape[0]
    if peak_x > correlation.shape[1] // 2:
        peak_x -= correlation.shape[1]
    return float(peak_y * downsample), float(peak_x * downsample)


def flow_sequence_features(history_ir: np.ndarray, *, downsample: int = 4) -> np.ndarray:
    """Return dy, dx, speed, sin(direction), cos(direction) for consecutive frames."""
    if history_ir.ndim != 3 or history_ir.shape[0] < 2:
        raise ValueError("IR history must have [time, height, width] with at least two frames")
    records = []
    for first, second in zip(history_ir[:-1], history_ir[1:], strict=True):
        dy, dx = phase_correlation_translation(first, second, downsample=downsample)
        speed = float(np.hypot(dy, dx))
        angle = float(np.arctan2(dy, dx))
        records.extend((dy, dx, speed, float(np.sin(angle)), float(np.cos(angle))))
    return np.asarray(records, dtype=np.float64)


def build_flow_feature_matrix(
    samples: pd.DataFrame,
    zarr_path: str,
    *,
    ir_channel: int = 2,
    downsample: int = 4,
    time_chunk: int = 12,
) -> np.ndarray:
    """Build motion features with one Zarr read per site/time chunk."""
    if time_chunk < 1:
        raise ValueError("time_chunk must be positive")
    frames = zarr.open_group(zarr_path, mode="r")["frames"]
    cache: dict[tuple[int, int, int], tuple[float, float]] = {}
    requested: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for row in samples.itertuples(index=False):
        site_index = int(row.site_index)
        indices = [int(value) for value in row.satellite_frame_indices]
        for first_index, second_index in zip(indices[:-1], indices[1:], strict=True):
            requested.setdefault((site_index, first_index // time_chunk), set()).add(
                (first_index, second_index)
            )
    for (site_index, _chunk_index), pairs in sorted(requested.items()):
        start = min(first for first, _second in pairs)
        stop = max(second for _first, second in pairs) + 1
        block = np.asarray(frames[site_index, start:stop, ir_channel], dtype=np.float32)
        for first_index, second_index in pairs:
            cache[(site_index, first_index, second_index)] = phase_correlation_translation(
                block[first_index - start], block[second_index - start],
                downsample=downsample,
            )
    rows = []
    for row in samples.itertuples(index=False):
        site_index = int(row.site_index)
        indices = [int(value) for value in row.satellite_frame_indices]
        features = []
        for first_index, second_index in zip(indices[:-1], indices[1:], strict=True):
            key = (site_index, first_index, second_index)
            dy, dx = cache[key]
            speed = float(np.hypot(dy, dx))
            angle = float(np.arctan2(dy, dx))
            features.extend((dy, dx, speed, float(np.sin(angle)), float(np.cos(angle))))
        rows.append(features)
    return np.asarray(rows, dtype=np.float64)


def cached_flow_feature_matrix(
    samples: pd.DataFrame,
    zarr_path: str,
    cache_dir: Path,
    *,
    ir_channel: int = 2,
    downsample: int = 4,
    time_chunk: int = 12,
) -> tuple[np.ndarray, str, bool]:
    """Load or atomically create a sample-keyed motion feature cache."""
    keys = [
        [int(row.site_index), [int(value) for value in row.satellite_frame_indices]]
        for row in samples.itertuples(index=False)
    ]
    identity = {
        "keys": keys,
        "ir_channel": ir_channel,
        "downsample": downsample,
        "time_chunk": time_chunk,
        "zarr_path": str(Path(zarr_path).resolve()),
    }
    cache_key = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"flow-{cache_key}.npy"
    if path.exists():
        values = np.load(path, allow_pickle=False)
        if len(values) != len(samples):
            raise ValueError("cached flow feature row count does not match samples")
        return values, cache_key, True
    values = build_flow_feature_matrix(
        samples, zarr_path, ir_channel=ir_channel, downsample=downsample,
        time_chunk=time_chunk,
    )
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, values, allow_pickle=False)
    temporary.replace(path)
    return values, cache_key, False


@dataclass
class MaskedRidge:
    """Independent closed-form ridge regressions for masked forecast horizons."""

    alpha: float = 1.0

    def fit(self, features: np.ndarray, targets: np.ndarray,
            mask: np.ndarray) -> MaskedRidge:
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if features.ndim != 2 or targets.shape != mask.shape or len(features) != len(targets):
            raise ValueError("incompatible ridge feature, target, or mask shapes")
        self.mean_ = features.mean(axis=0)
        self.scale_ = features.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        normalized = (features - self.mean_) / self.scale_
        design = np.column_stack((np.ones(len(normalized)), normalized))
        weights = []
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        for horizon in range(targets.shape[1]):
            valid = mask[:, horizon].astype(bool)
            if not np.any(valid):
                raise ValueError("every horizon needs at least one valid target")
            x, y = design[valid], targets[valid, horizon]
            weights.append(np.linalg.lstsq(x.T @ x + penalty, x.T @ y, rcond=None)[0])
        self.weights_ = np.column_stack(weights)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not hasattr(self, "weights_"):
            raise ValueError("ridge model must be fitted before prediction")
        normalized = (features - self.mean_) / self.scale_
        return np.column_stack((np.ones(len(normalized)), normalized)) @ self.weights_
