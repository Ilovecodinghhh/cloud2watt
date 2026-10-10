import numpy as np
import pandas as pd
import pytest
import torch

from cloud2watt.models import PowerMLP, PowerTCN, build_power_model
from cloud2watt.training import (
    FeatureStatistics,
    PowerForecastDataset,
    create_loader,
    load_checkpoint,
    masked_mae_loss,
    run_epoch,
    save_checkpoint,
    seed_everything,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    times = pd.date_range("2021-01-01", periods=16, freq="15min", tz="UTC")
    power = pd.DataFrame({
        "normalized_power": np.linspace(0.1, 0.8, 16),
        "solar_elevation_deg": np.linspace(1, 30, 16),
        "solar_azimuth_deg": np.linspace(100, 200, 16),
        "clear_sky_ghi_wm2": np.linspace(10, 500, 16),
    })
    sites = pd.DataFrame({
        "ss_id": ["a"], "kWp": [4.0], "tilt": [30.0], "orientation": [180.0],
        "latitude_rounded": [51.0], "longitude_rounded": [-1.0],
    })
    samples = pd.DataFrame({
        "site_id": ["a", "a"], "issue_time_utc": [times[3], times[4]],
        "power_history_row_indices": [np.arange(4), np.arange(1, 5)],
        "target_row_indices": [np.arange(4, 10), np.arange(5, 11)],
        "target_mask": [np.ones(6, dtype=bool), np.ones(6, dtype=bool)],
    })
    return samples, power, sites


def test_models_have_expected_shape_and_parameter_budget() -> None:
    inputs = (torch.zeros(2, 4), torch.zeros(2, 6, 3), torch.zeros(2, 5))
    for model in (PowerMLP(), PowerTCN()):
        assert model(*inputs).shape == (2, 6)
        assert sum(parameter.numel() for parameter in model.parameters()) < 1_000_000


def test_model_factory_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="unsupported model"):
        build_power_model("transformer")


def test_masked_mae_ignores_invalid_targets() -> None:
    prediction, target = torch.tensor([[1.0, 100.0]]), torch.tensor([[0.0, 0.0]])
    assert masked_mae_loss(prediction, target, torch.tensor([[True, False]])).item() == 1.0


def test_statistics_are_fitted_from_supplied_training_rows_only() -> None:
    samples, power, sites = _frames()
    first = FeatureStatistics.fit(samples.iloc[[0]], power, sites)
    modified = power.copy()
    modified.loc[10:, "clear_sky_ghi_wm2"] = 1_000_000
    assert FeatureStatistics.fit(samples.iloc[[0]], modified, sites) == first


def test_dataset_exposes_only_approved_inputs() -> None:
    samples, power, sites = _frames()
    stats = FeatureStatistics.fit(samples.iloc[[0]], power, sites)
    item = PowerForecastDataset(samples, power, sites, stats)[0]
    assert set(item) == {"power_history", "solar_future", "site", "target", "target_mask",
                         "site_id", "issue_time_utc", "target_solar_elevation_deg"}
    assert item["power_history"].shape == (4,)
    assert item["solar_future"].shape == (6, 3)


def test_cpu_smoke_training_and_checkpoint_restore(tmp_path) -> None:
    seed_everything(42)
    samples, power, sites = _frames()
    stats = FeatureStatistics.fit(samples, power, sites)
    dataset = PowerForecastDataset(samples, power, sites, stats)
    loader = create_loader(dataset, batch_size=2, shuffle=False, seed=42)
    model = PowerMLP(hidden_size=16, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = run_epoch(model, loader, device=torch.device("cpu"), optimizer=optimizer)
    assert np.isfinite(loss)
    inputs = tuple(torch.stack([dataset[0][key]]) for key in
                   ("power_history", "solar_future", "site"))
    expected = model(*inputs).detach()
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, optimizer, epoch=1, validation_loss=loss,
                    statistics=stats, metadata={"split": "train"})
    restored = PowerMLP(hidden_size=16, dropout=0.0)
    payload = load_checkpoint(path, restored)
    torch.testing.assert_close(restored(*inputs).detach(), expected)
    assert payload["feature_statistics_sha256"] == stats.sha256
