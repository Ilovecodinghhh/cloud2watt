"""Compact power-only neural forecasting models."""

from __future__ import annotations

import torch
from torch import nn


class PowerMLP(nn.Module):
    """Forecast all horizons from flattened power, solar, and site features."""

    def __init__(self, *, history_steps: int = 4, solar_features: int = 3,
                 site_features: int = 5, forecast_steps: int = 6,
                 hidden_size: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        input_size = history_steps + forecast_steps * solar_features + site_features
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, forecast_steps),
        )

    def forward(self, power_history: torch.Tensor, solar_future: torch.Tensor,
                site: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((power_history, solar_future.flatten(1), site), dim=1))


class _CausalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        transformed = torch.relu(self.conv(nn.functional.pad(values, (self.padding, 0))))
        return values + self.dropout(transformed)


class PowerTCN(nn.Module):
    """Small causal temporal convolution model for recent power history."""

    def __init__(self, *, solar_features: int = 3, site_features: int = 5,
                 forecast_steps: int = 6, channels: int = 64,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(1, channels, kernel_size=1)
        self.temporal = nn.Sequential(
            _CausalBlock(channels, 1, dropout), _CausalBlock(channels, 2, dropout)
        )
        self.head = nn.Sequential(
            nn.Linear(channels + forecast_steps * solar_features + site_features, channels),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(channels, forecast_steps),
        )

    def forward(self, power_history: torch.Tensor, solar_future: torch.Tensor,
                site: torch.Tensor) -> torch.Tensor:
        encoded = self.temporal(self.input_projection(power_history.unsqueeze(1)))[:, :, -1]
        return self.head(torch.cat((encoded, solar_future.flatten(1), site), dim=1))


def build_power_model(name: str, **kwargs: object) -> nn.Module:
    """Create a supported power-only model by configuration name."""
    models = {"mlp": PowerMLP, "tcn": PowerTCN}
    if name not in models:
        raise ValueError(f"unsupported model: {name}")
    return models[name](**kwargs)
