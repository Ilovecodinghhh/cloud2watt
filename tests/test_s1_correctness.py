"""Regression tests for the S0 audit findings, using numerical reference behavior."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from cloud2watt.epoch import evaluate, fit_epochs, train_epoch
from cloud2watt.evaluation.forecast import (
    PROTOCOL,
    ForecastError,
    require_primary,
    score_forecast,
)
from cloud2watt.evaluation.metrics import masked_mae
from cloud2watt.models import PowerMLP, SharedSatelliteEncoder
from cloud2watt.run_state import (
    REQUIRED_ARTIFACTS,
    atomic_json,
    atomic_path,
    is_complete,
    load_epoch,
    mark_complete,
    run_lock,
)
from cloud2watt.training import masked_mae_loss, seed_everything


def _keys(n):
    return pd.DataFrame({"site_id": ["a"] * n,
                         "issue_time_utc": pd.date_range("2021-01-01", periods=n,
                                                        freq="15min", tz="UTC")})


def test_daylight_primary_uses_three_equal_horizons_and_strict_solar_boundary():
    y = np.zeros((3, 6))
    p = np.array([[99, 1, 3, 6, 99, 99], [99, 2, 99, 99, 99, 99],
                  [99, 99, 99, 99, 99, 99]], float)
    m = np.ones((3, 6), bool)
    e = np.array([[20] * 6, [5, 20, 5, 5, 5, 5], [0] * 6], float)
    r = score_forecast(p, y, m, e, _keys(3))
    assert require_primary(r) == pytest.approx((1.5 + 3 + 6) / 3)
    assert r["daylight"][1]["valid_targets"] == 2
    assert r["solar_support"]["low_sun"][2] == 1
    assert r["all_day"][1]["mae"] != r["daylight"][1]["mae"]
    e[:, 3] = 0
    with pytest.raises(ValueError, match="primary horizons"):
        require_primary(score_forecast(p, y, m, e, _keys(3)))


def test_invalid_prediction_never_shrinks_population_even_at_night():
    y = np.zeros((2, 6))
    p, m, e = y.copy(), np.ones_like(y, bool), y.copy()
    p[0, 2] = np.inf
    with pytest.raises(ForecastError) as exc:
        score_forecast(p, y, m, e, _keys(2))
    assert exc.value.report["failure_rate"] == pytest.approx(1 / 12)
    assert exc.value.report["keys"][0]["site_id"] == "a"
    with pytest.raises(ValueError, match="non-finite"):
        masked_mae(y, p, m)
    m[0, 2] = False
    assert masked_mae(y, p, m) == 0
    # Mask before subtracting: NaNs at ignored labels must not poison gradients.
    pred = torch.tensor([1., float("nan")], requires_grad=True)
    loss = masked_mae_loss(pred, torch.tensor([0., float("nan")]), torch.tensor([1, 0]))
    loss.backward()
    torch.testing.assert_close(pred.grad, torch.tensor([1., 0.]))


def test_comparison_keys_are_order_invariant_and_duplicate_safe():
    y = np.arange(18).reshape(3, 6) / 100
    m, e, keys = np.ones_like(y, bool), np.ones_like(y) * 20, _keys(3)
    a = score_forecast(y, y, m, e, keys)
    ix = [2, 0, 1]
    b = score_forecast(y[ix], y[ix], m[ix], e[ix], keys.iloc[ix])
    assert a["comparison"] == b["comparison"]
    keys.iloc[1] = keys.iloc[0]
    with pytest.raises(ValueError, match="duplicate"):
        score_forecast(y, y, m, e, keys)


@pytest.mark.parametrize("mask", [[1, 1, 1, 1], [0, 1, 1, 1], [1, 0, 1, 1],
                                  [1, 1, 1, 0], [0, 0, 0, 0]])
def test_masked_recurrence_matches_only_valid_frames(mask):
    torch.manual_seed(7)
    encoder = SharedSatelliteEncoder(width=8, embedding_size=16).eval()
    x = torch.randn(1, 4, 3, 16, 16)
    m = torch.tensor([mask], dtype=torch.bool)
    expected = torch.zeros(1, 16)
    if any(mask):
        valid = x[:, m[0]]
        expected = encoder(valid, torch.ones(1, valid.shape[1], dtype=torch.bool))
    x[~m] = float("nan")
    torch.testing.assert_close(encoder(x, m), expected, atol=1e-7, rtol=1e-6)


class TinyPower(Dataset):
    def __init__(self, n=7):
        generator = torch.Generator().manual_seed(81)
        self.x = torch.randn(n, 4, generator=generator)
        self.y = torch.rand(n, 6, generator=generator)
        self.mask = torch.ones(n, 6, dtype=torch.bool)
        self.mask[::2, 1:] = False
        self.keys = _keys(n)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"power_history": self.x[i], "solar_future": torch.zeros(6, 3),
                "site": torch.zeros(5), "target": self.y[i], "target_mask": self.mask[i],
                "target_solar_elevation_deg": torch.ones(6) * 20,
                "site_id": "a", "issue_time_utc": self.keys.iloc[i].issue_time_utc.isoformat()}


def test_accumulation_equals_large_batch_with_unequal_masks_and_partial_final_group():
    seed_everything(19)
    a = PowerMLP(hidden_size=8, dropout=0)
    b = copy.deepcopy(a)
    oa, ob = torch.optim.SGD(a.parameters(), lr=.01), torch.optim.SGD(b.parameters(), lr=.01)
    data = TinyPower()
    train_epoch(a, DataLoader(data, batch_size=6), device=torch.device("cpu"), optimizer=oa)
    train_epoch(b, DataLoader(data, batch_size=2), device=torch.device("cpu"), optimizer=ob,
                accumulation_steps=3)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa, pb, atol=1e-7, rtol=1e-6)


def _training_setup(workers=0):
    seed_everything(42)
    model = PowerMLP(hidden_size=8, dropout=.25)
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    data = TinyPower()
    loaders = {key: DataLoader(data, batch_size=2, shuffle=key == "train", num_workers=workers,
                              generator=torch.Generator().manual_seed(90 + i))
               for i, key in enumerate(("train", "validation"))}
    return model, optimizer, loaders


@pytest.mark.parametrize("workers", [0, 2])
def test_epoch_resume_replays_dropout_shuffle_optimizer_and_history(tmp_path, workers):
    full, paused = tmp_path / "full", tmp_path / "paused"
    full.mkdir()
    paused.mkdir()
    identity = {"protocol": PROTOCOL, "case": "resume"}
    options = {"device": torch.device("cpu"), "identity": identity, "max_epochs": 3,
               "patience": 5, "accumulation_steps": 2}
    model, optimizer, loaders = _training_setup(workers)
    assert fit_epochs(model, optimizer, loaders, output=full, **options)
    expected, _ = evaluate(model, loaders["validation"], torch.device("cpu"))
    model, optimizer, loaders = _training_setup(workers)
    assert not fit_epochs(model, optimizer, loaders, output=paused, stop_after_epoch=1, **options)
    assert not (paused / "complete.json").exists()
    # New model/optimizer/loaders, as on process restart; restored state must win.
    model, optimizer, loaders = _training_setup(workers)
    torch.rand(100)
    assert fit_epochs(model, optimizer, loaders, output=paused, resume=True, **options)
    actual, _ = evaluate(model, loaders["validation"], torch.device("cpu"))
    assert actual == expected
    a, b = load_epoch(full / "latest.pt", identity), load_epoch(paused / "latest.pt", identity)
    assert a["history"] == b["history"]
    for k in a["model_state"]:
        torch.testing.assert_close(a["model_state"][k], b["model_state"][k], rtol=0, atol=0)
    with pytest.raises(ValueError, match="identity mismatch"):
        load_epoch(paused / "latest.pt", identity | {"config_hash": "changed"})


def test_selection_uses_daylight_score_and_retains_earlier_tie(tmp_path, monkeypatch):
    model, optimizer, loaders = _training_setup()
    scores = iter([.3, .1, .1])
    monkeypatch.setattr("cloud2watt.epoch.evaluate",
                        lambda *a, **kw: ({"daylight_primary_mae": next(scores)}, None))
    fit_epochs(model, optimizer, loaders, device=torch.device("cpu"), identity={}, output=tmp_path,
               max_epochs=3, patience=5)
    assert torch.load(tmp_path / "best.pt", weights_only=True)["epoch"] == 2


def test_atomic_write_preserves_old_file_on_failure_and_completion_detects_corruption(tmp_path):
    target = tmp_path / "metrics.json"
    atomic_json(target, {"old": True})
    with pytest.raises(RuntimeError), atomic_path(target) as temporary:
        temporary.write_text("partial", encoding="utf-8")
        raise RuntimeError("interrupted writer")
    assert json.loads(target.read_text()) == {"old": True}
    identity = {"protocol": PROTOCOL, "config_hash": "a"}
    assert not is_complete(tmp_path, identity)
    with pytest.raises(FileNotFoundError):
        mark_complete(tmp_path, identity)
    assert not (tmp_path / "complete.json").exists()
    for name in REQUIRED_ARTIFACTS:
        (tmp_path / name).write_bytes(b"fixture")
    mark_complete(tmp_path, identity)
    assert is_complete(tmp_path, identity)
    assert not is_complete(tmp_path, identity | {"config_hash": "b"})
    target.write_bytes(b"corrupt")
    assert not is_complete(tmp_path, identity)


def test_run_lock_rejects_concurrent_writer_and_releases(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="active"), run_lock(tmp_path):
            pass
    with run_lock(tmp_path):
        pass


def test_matrix_ranks_daylight_metric_and_excludes_incomplete_seed_sets(tmp_path):
    from cloud2watt.experiment_matrix import rank_completed_runs
    paths = []
    for mode, scores in (("full", [.3, .3, .3]), ("power_solar", [.1, .1, .1]),
                         ("satellite_solar", [.01])):
        for seed, score in zip((42, 123, 2026), scores, strict=False):
            path = tmp_path / f"{mode}-{seed}"
            path.mkdir()
            identity = {"mode": mode, "seed": seed, "scope": "development", "parameters": 10,
                        **dict.fromkeys(("config_hash", "source_hash", "environment_hash",
                                         "local_data_hash", "split_manifest_hash"), "same")}
            for name in REQUIRED_ARTIFACTS:
                (path / name).write_bytes(b"fixture")
            atomic_json(path / "provenance.json", {"identity": identity})
            atomic_json(path / "metrics.json", {"validation_development": {"none": {
                "daylight_primary_mae": score, "comparison": {"keys": "same"},
                "all_day_mae": 1 - score}}})
            mark_complete(path, identity)
            paths.append(path)
    ranked = rank_completed_runs(paths)["candidates"]
    assert [r["mode"] for r in ranked] == ["power_solar", "full", "satellite_solar"]
    assert not ranked[-1]["selection_ready"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_amp_scaler_has_finite_updates_and_restorable_state():
    model = PowerMLP(hidden_size=8, dropout=0).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    scaler = torch.amp.GradScaler("cuda")
    train_epoch(model, DataLoader(TinyPower(), batch_size=2), device=torch.device("cuda"),
                optimizer=optimizer, accumulation_steps=2, use_amp=True, scaler=scaler)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    other = torch.amp.GradScaler("cuda")
    other.load_state_dict(scaler.state_dict())
    assert other.state_dict() == scaler.state_dict()
