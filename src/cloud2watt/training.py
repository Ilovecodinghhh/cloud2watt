"""Leakage-safe datasets and training utilities for power-only models."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cloud2watt.data.paired import FORECAST_MINUTES

SOLAR_COLUMNS = ("solar_elevation_deg", "solar_azimuth_deg", "clear_sky_ghi_wm2")
SITE_COLUMNS = ("kWp", "tilt", "orientation", "latitude_rounded", "longitude_rounded")


@dataclass(frozen=True)
class FeatureStatistics:
    """Training-only mean and scale for continuous inputs."""

    solar_mean: list[float]
    solar_scale: list[float]
    site_mean: list[float]
    site_scale: list[float]

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def fit(cls, samples: pd.DataFrame, power: pd.DataFrame,
            sites: pd.DataFrame) -> FeatureStatistics:
        """Fit statistics from the explicitly supplied training samples only."""
        if samples.empty:
            raise ValueError("normalization samples must not be empty")
        indices = np.concatenate(samples["target_row_indices"].map(np.asarray).to_list())
        indices = np.unique(indices[indices >= 0].astype(int))
        solar_mean, solar_scale = _mean_scale(
            power.iloc[indices].loc[:, SOLAR_COLUMNS].to_numpy(dtype=np.float64)
        )
        site_ids = samples["site_id"].astype(str).unique()
        site_values = sites.loc[
            sites["ss_id"].astype(str).isin(site_ids), SITE_COLUMNS
        ].to_numpy(dtype=np.float64)
        site_mean, site_scale = _mean_scale(site_values)
        return cls(solar_mean.tolist(), solar_scale.tolist(),
                   site_mean.tolist(), site_scale.tolist())


def _mean_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0)
    scale = np.nanstd(values, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return mean, scale


class PowerForecastDataset(Dataset):
    """Materialize only fields allowed by the power-only input contract."""

    def __init__(self, samples: pd.DataFrame, power: pd.DataFrame,
                 sites: pd.DataFrame, statistics: FeatureStatistics) -> None:
        power_values = power["normalized_power"].to_numpy(dtype=np.float32)
        solar_values = power.loc[:, SOLAR_COLUMNS].to_numpy(dtype=np.float32)
        site_lookup = sites.assign(_site_id=sites["ss_id"].astype(str)).set_index("_site_id")
        solar_mean = np.asarray(statistics.solar_mean, dtype=np.float32)
        solar_scale = np.asarray(statistics.solar_scale, dtype=np.float32)
        site_mean = np.asarray(statistics.site_mean, dtype=np.float32)
        site_scale = np.asarray(statistics.site_scale, dtype=np.float32)
        self.records: list[dict[str, Any]] = []
        for row in samples.itertuples(index=False):
            history_index = np.asarray(row.power_history_row_indices, dtype=int)
            target_index = np.asarray(row.target_row_indices, dtype=int)
            safe_index = np.maximum(target_index, 0)
            mask = np.asarray(row.target_mask, dtype=bool) & (target_index >= 0)
            target = power_values[safe_index].copy()
            target[~mask] = 0.0
            solar = np.nan_to_num((solar_values[safe_index] - solar_mean) / solar_scale)
            site = site_lookup.loc[str(row.site_id), list(SITE_COLUMNS)].to_numpy(dtype=np.float32)
            site = np.nan_to_num((site - site_mean) / site_scale)
            self.records.append({
                "power_history": power_values[history_index], "solar_future": solar,
                "site": site, "target": target, "target_mask": mask,
                "site_id": str(row.site_id),
                "issue_time_utc": pd.Timestamp(row.issue_time_utc).isoformat(),
            })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            key: (value if key in {"site_id", "issue_time_utc"}
                  else torch.as_tensor(value, dtype=torch.float32))
            for key, value in self.records[index].items()
        }


def masked_mae_loss(prediction: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over valid targets only."""
    valid = mask.bool()
    if not torch.any(valid):
        raise ValueError("masked loss requires at least one valid target")
    return torch.abs(prediction - target)[valid].mean()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_loader(dataset: Dataset, *, batch_size: int, shuffle: bool,
                  seed: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed))


def run_epoch(model: nn.Module, loader: DataLoader, *, device: torch.device,
              optimizer: torch.optim.Optimizer | None = None) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss, total_targets = 0.0, 0
    with torch.enable_grad() if training else torch.no_grad():
        for batch in loader:
            prediction = model(batch["power_history"].to(device),
                               batch["solar_future"].to(device), batch["site"].to(device))
            target = batch["target"].to(device)
            mask = batch["target_mask"].to(device).bool()
            loss = masked_mae_loss(prediction, target, mask)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            count = int(mask.sum())
            total_loss += float(loss.detach()) * count
            total_targets += count
    if total_targets == 0:
        raise ValueError("epoch contains no valid targets")
    return total_loss / total_targets


def predict(model: nn.Module, loader: DataLoader,
            device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions, targets, masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["power_history"].to(device),
                           batch["solar_future"].to(device), batch["site"].to(device))
            predictions.append(output.cpu().numpy())
            targets.append(batch["target"].numpy())
            masks.append(batch["target_mask"].numpy().astype(bool))
    return np.concatenate(predictions), np.concatenate(targets), np.concatenate(masks)


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    *, epoch: int, validation_loss: float,
                    statistics: FeatureStatistics, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "epoch": epoch, "validation_loss": validation_loss,
                "feature_statistics": asdict(statistics),
                "feature_statistics_sha256": statistics.sha256,
                "metadata": metadata}, path)


def load_checkpoint(path: Path, model: nn.Module,
                    optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def metrics_by_horizon(prediction: np.ndarray, target: np.ndarray,
                       mask: np.ndarray) -> list[dict[str, float | int]]:
    rows = []
    for index, horizon in enumerate(FORECAST_MINUTES):
        valid = mask[:, index]
        error = prediction[valid, index] - target[valid, index]
        rows.append({"horizon_minutes": horizon, "mae": float(np.mean(np.abs(error))),
                     "rmse": float(np.sqrt(np.mean(error**2))),
                     "valid_targets": int(valid.sum())})
    return rows
