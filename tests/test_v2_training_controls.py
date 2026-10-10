"""Numerical tests for weighted objectives and explicit minimum training budgets."""
import copy

import numpy as np
import pytest
import torch
from test_s1_correctness import TinyPower, _training_setup
from torch.utils.data import DataLoader

from cloud2watt.epoch import fit_epochs, train_epoch
from cloud2watt.models import PowerMLP


class MixedDay(TinyPower):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        elevation = torch.ones(6) * 20
        if index % 3 == 0:
            elevation[:] = -5
        if index % 3 == 1:
            elevation[2] = 5  # Strict daylight boundary, one unsupported horizon.
        item["target_solar_elevation_deg"] = elevation
        return item


def test_weighted_daylight_accumulation_matches_effective_batch_and_supervises_all_heads():
    torch.manual_seed(10)
    a = PowerMLP(hidden_size=8, dropout=0)
    b = copy.deepcopy(a)
    before = {k: v.clone() for k, v in a.state_dict().items()}
    data = MixedDay(7)
    for model, batch, accumulation in ((a, 6, 1), (b, 2, 3)):
        train_epoch(model, DataLoader(data, batch_size=batch), device=torch.device("cpu"),
                    optimizer=torch.optim.SGD(model.parameters(), lr=.01),
                    accumulation_steps=accumulation, primary_loss_weight=.8)
    for key in a.state_dict():
        torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], atol=1e-7, rtol=1e-6)
    # The six output bias entries must all receive supervision, including secondary horizons.
    bias_key = [k for k in before if before[k].shape == (6,)][-1]
    assert (before[bias_key] != a.state_dict()[bias_key]).all()


def test_night_only_weight_is_not_renormalized():
    class Night(TinyPower):
        def __getitem__(self, i):
            row = super().__getitem__(i)
            row["target_solar_elevation_deg"][:] = 0
            return row

    torch.manual_seed(4)
    a = PowerMLP(hidden_size=8, dropout=0)
    b = copy.deepcopy(a)
    loader = DataLoader(Night(), batch_size=7)
    # Learning rates absorb the 0.2 multiplier, with no gradient clipping activation here.
    for model, rate, weight in ((a, .001, .8), (b, .0002, 0)):
        train_epoch(model, loader, device=torch.device("cpu"),
                    optimizer=torch.optim.SGD(model.parameters(), lr=rate),
                    primary_loss_weight=weight)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa, pb, atol=1e-7, rtol=1e-6)


def test_minimum_epochs_prevents_premature_stop_and_keeps_earlier_best(tmp_path, monkeypatch):
    model, optimizer, loaders = _training_setup()
    scores = iter([.1, .2, .3, .4, .5])
    monkeypatch.setattr("cloud2watt.epoch.evaluate",
                        lambda *a, **kw: ({"daylight_primary_mae": next(scores)}, None))
    fit_epochs(model, optimizer, loaders, device=torch.device("cpu"), identity={}, output=tmp_path,
               max_epochs=5, min_epochs=4, patience=1)
    payload = torch.load(tmp_path / "latest.pt", weights_only=True)
    assert payload["epoch"] == 4 and payload["best_epoch"] == 1
    assert np.isfinite(payload["best_score"])
    with pytest.raises(ValueError, match="min_epochs"):
        fit_epochs(model, optimizer, loaders, device=torch.device("cpu"), identity={},
                   output=tmp_path, max_epochs=2, min_epochs=3, patience=1)
