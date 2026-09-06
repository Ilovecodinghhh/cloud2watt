import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from cloud2watt.models import SatelliteLateFusion, SharedSatelliteEncoder
from cloud2watt.satellite_training import predict_satellite, run_satellite_epoch
from scripts.run_cnn_experiment_matrix import completed_formal_run


def _inputs(batch: int = 2) -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(batch, 4, 3, 32, 32), torch.ones(batch, 4, dtype=torch.bool),
        torch.randn(batch, 4), torch.randn(batch, 6, 3), torch.randn(batch, 5),
    )


class _TinyDataset(Dataset):
    def __init__(self) -> None:
        satellite, satellite_mask, power, solar, site = _inputs(4)
        self.values = {
            "satellite": satellite, "satellite_mask": satellite_mask,
            "power_history": power, "solar_future": solar, "site": site,
            "target": torch.rand(4, 6), "target_mask": torch.ones(4, 6),
        }

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.values.items()}


def test_shared_encoder_applies_one_frame_encoder_to_all_timesteps() -> None:
    encoder = SharedSatelliteEncoder(width=8, embedding_size=16)
    satellite, mask, *_rest = _inputs()
    assert encoder(satellite, mask).shape == (2, 16)
    convolution_weights = [parameter for name, parameter in encoder.named_parameters()
                           if name.endswith("frame_encoder.0.weight")]
    assert len(convolution_weights) == 1


@pytest.mark.parametrize("mode", sorted(SatelliteLateFusion.MODES))
def test_fusion_modes_have_expected_shape_and_budget(mode: str) -> None:
    model = SatelliteLateFusion(mode=mode, cnn_width=8, embedding_size=16, dropout=0.0)
    assert model(*_inputs()).shape == (2, 6)
    assert sum(parameter.numel() for parameter in model.parameters()) < 10_000_000


def test_satellite_zero_and_shuffle_controls_are_deterministic() -> None:
    model = SatelliteLateFusion(cnn_width=8, embedding_size=16, dropout=0.0).eval()
    inputs = _inputs()
    zero_a = model(*inputs, satellite_perturbation="zero")
    zero_b = model(*inputs, satellite_perturbation="zero")
    shuffled = model(*inputs, satellite_perturbation="shuffle")
    torch.testing.assert_close(zero_a, zero_b)
    assert not torch.equal(shuffled, model(*inputs))


def test_branch_ablation_modes_ignore_excluded_inputs() -> None:
    inputs = list(_inputs())
    power_only = SatelliteLateFusion(
        mode="power_solar", cnn_width=8, embedding_size=16, dropout=0.0
    ).eval()
    expected = power_only(*inputs)
    inputs[0] = inputs[0] + 1000
    torch.testing.assert_close(power_only(*inputs), expected)
    satellite_only = SatelliteLateFusion(
        mode="satellite_solar", cnn_width=8, embedding_size=16, dropout=0.0
    ).eval()
    expected = satellite_only(*inputs)
    inputs[2] = inputs[2] + 1000
    torch.testing.assert_close(satellite_only(*inputs), expected)


def test_cpu_forward_backward_and_perturbed_prediction() -> None:
    loader = DataLoader(_TinyDataset(), batch_size=2)
    model = SatelliteLateFusion(cnn_width=8, embedding_size=16, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = run_satellite_epoch(
        model, loader, device=torch.device("cpu"), optimizer=optimizer,
        accumulation_steps=2,
    )
    assert np.isfinite(loss)
    prediction, target, mask = predict_satellite(
        model, loader, torch.device("cpu"), perturbation="zero"
    )
    assert prediction.shape == target.shape == mask.shape == (4, 6)


def test_checkpoint_state_round_trip_preserves_prediction(tmp_path) -> None:
    model = SatelliteLateFusion(cnn_width=8, embedding_size=16, dropout=0.0).eval()
    inputs = _inputs()
    expected = model(*inputs)
    path = tmp_path / "fusion.pt"
    torch.save(model.state_dict(), path)
    restored = SatelliteLateFusion(cnn_width=8, embedding_size=16, dropout=0.0).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(restored(*inputs), expected)


def test_invalid_mode_and_perturbation_are_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported fusion mode"):
        SatelliteLateFusion(mode="future_power")
    model = SatelliteLateFusion(cnn_width=8, embedding_size=16)
    with pytest.raises(ValueError, match="unsupported satellite perturbation"):
        model(*_inputs(), satellite_perturbation="rotate")


def test_matrix_only_skips_completed_uncapped_run(tmp_path) -> None:
    run = tmp_path / "cnn-late-fusion-v1-full-s42-deadbeef"
    run.mkdir()
    (run / "metrics.json").write_text("{}\n", encoding="utf-8")
    provenance = {
        "mode": "full", "seed": 42, "max_epochs": 30,
        "max_train_samples": None, "max_validation_samples": None,
        "test_unlocked": False,
    }
    (run / "provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    assert completed_formal_run(tmp_path, "cnn-late-fusion-v1", "full", 42, 30)
    provenance["max_train_samples"] = 8
    (run / "provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    assert not completed_formal_run(tmp_path, "cnn-late-fusion-v1", "full", 42, 30)
