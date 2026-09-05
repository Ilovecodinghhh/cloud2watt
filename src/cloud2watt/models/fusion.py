"""Lightweight shared-CNN late-fusion model for PV nowcasting."""

from __future__ import annotations

import torch
from torch import nn


class SharedSatelliteEncoder(nn.Module):
    """Encode every satellite frame with the same compact 2-D CNN."""

    def __init__(self, in_channels: int = 3, width: int = 32, embedding_size: int = 96) -> None:
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(in_channels, width, 5, stride=2, padding=2),
            nn.GroupNorm(4, width), nn.SiLU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, width * 2), nn.SiLU(),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, width * 4), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(width * 4, embedding_size), nn.SiLU(),
        )
        self.temporal = nn.GRU(embedding_size, embedding_size, batch_first=True)

    def forward(self, satellite: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if satellite.ndim != 5:
            raise ValueError("satellite must have [batch, time, channel, height, width] shape")
        batch, steps, channels, height, width = satellite.shape
        encoded = self.frame_encoder(satellite.reshape(batch * steps, channels, height, width))
        encoded = encoded.reshape(batch, steps, -1) * mask.float().unsqueeze(-1)
        output, _state = self.temporal(encoded)
        lengths = mask.long().sum(dim=1).clamp(min=1) - 1
        return output[torch.arange(batch, device=output.device), lengths]


class SatelliteLateFusion(nn.Module):
    """Fuse satellite, recent power, future solar, and static site embeddings."""

    MODES = {"full", "satellite_solar", "power_solar"}

    def __init__(self, *, mode: str = "full", cnn_width: int = 32,
                 embedding_size: int = 96, dropout: float = 0.1,
                 forecast_steps: int = 6, solar_features: int = 3,
                 site_features: int = 5) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unsupported fusion mode: {mode}")
        self.mode = mode
        self.satellite_encoder = SharedSatelliteEncoder(
            width=cnn_width, embedding_size=embedding_size
        ) if mode != "power_solar" else None
        self.power_encoder = nn.Sequential(
            nn.Linear(4, embedding_size), nn.SiLU(), nn.Dropout(dropout)
        ) if mode != "satellite_solar" else None
        self.context_encoder = nn.Sequential(
            nn.Linear(forecast_steps * solar_features + site_features, embedding_size),
            nn.SiLU(), nn.Dropout(dropout),
        )
        branch_count = 1 + int(self.satellite_encoder is not None) + int(
            self.power_encoder is not None
        )
        self.head = nn.Sequential(
            nn.Linear(branch_count * embedding_size, embedding_size * 2),
            nn.SiLU(), nn.Dropout(dropout), nn.Linear(embedding_size * 2, forecast_steps),
        )

    def forward(self, satellite: torch.Tensor, satellite_mask: torch.Tensor,
                power_history: torch.Tensor, solar_future: torch.Tensor,
                site: torch.Tensor, *, satellite_perturbation: str = "none") -> torch.Tensor:
        if satellite_perturbation == "zero":
            satellite = torch.zeros_like(satellite)
            satellite_mask = torch.zeros_like(satellite_mask)
        elif satellite_perturbation == "shuffle":
            order = torch.roll(torch.arange(len(satellite), device=satellite.device), 1)
            satellite, satellite_mask = satellite[order], satellite_mask[order]
        elif satellite_perturbation != "none":
            raise ValueError(f"unsupported satellite perturbation: {satellite_perturbation}")
        branches = [self.context_encoder(torch.cat((solar_future.flatten(1), site), dim=1))]
        if self.power_encoder is not None:
            branches.append(self.power_encoder(power_history))
        if self.satellite_encoder is not None:
            branches.append(self.satellite_encoder(satellite, satellite_mask))
        return self.head(torch.cat(branches, dim=1))
