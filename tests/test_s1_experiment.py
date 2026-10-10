"""Synthetic source-to-training acceptance of V2 identity and matrix behavior."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
import zarr

from cloud2watt.data.paired import build_sample_index
from cloud2watt.evaluation.forecast import PROTOCOL
from cloud2watt.experiment import execute, parser_for, prepare
from cloud2watt.run_state import is_complete, json_hash


@pytest.fixture
def synthetic_run(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    times = pd.date_range("2021-01-01", periods=10 * 96, freq="15min", tz="UTC")
    sites = pd.DataFrame({"ss_id": ["a", "b", "c", "d"], "kWp": [2., 3., 4., 5.],
                          "tilt": [30.] * 4, "orientation": [180.] * 4,
                          "latitude_rounded": [50., 51., 52., 53.],
                          "longitude_rounded": [-1.] * 4})
    power = pd.concat([pd.DataFrame({"ss_id": site, "datetime_GMT": times,
                                    "normalized_power": .3 + .1 * np.sin(np.arange(len(times))),
                                    "is_valid": True, "solar_elevation_deg": 20.,
                                    "solar_azimuth_deg": 180., "clear_sky_ghi_wm2": 500.})
                       for site in sites.ss_id], ignore_index=True)
    # Coarse issue spacing keeps the fixture small; physical row references remain exact.
    samples, _ = build_sample_index(power, sites, issue_times=times[4:-16:8],
                                    satellite_times=times)
    for name, frame in (("power_15min", power), ("sites", sites), ("sample_index", samples)):
        frame.to_parquet(data / f"{name}.parquet", index=False)
    manifest = {"dataset_version": "synthetic-s1"}
    manifest["manifest_content_sha256"] = json_hash(manifest)
    (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    group = zarr.open_group(data / "satellite_frames.zarr", mode="w")
    group.create_array("frames", data=np.random.default_rng(7).normal(
        size=(4, len(times), 3, 8, 8)).astype("float32"), chunks=(1, 16, 3, 8, 8))
    group.create_array("time_ns", data=times.to_numpy(dtype="datetime64[ns]").astype("int64"))
    config = {"run_name": "synthetic-s1", "data_version": "synthetic-s1",
              "evaluation_protocol": PROTOCOL, "random_seeds": [42], "embargo_hours": 4,
              "model": {"mode": "full", "cnn_width": 8, "embedding_size": 16, "dropout": .1},
              "training": {"batch_size": 2, "num_workers": 0, "time_chunk": 12,
                           "learning_rate": .001, "max_epochs": 2, "patience": 4,
                           "device": "cpu", "accumulation_steps": 3},
              "satellite_statistics": {"maximum_frames": 12},
              "selection": {"metric": "daylight_primary_mae"}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    args = parser_for("satellite").parse_args([
        "--data", str(data), "--config", str(path), "--output-root", str(tmp_path / "runs"),
        "--max-train-samples", "7", "--max-validation-samples", "4", "--auto-resume"])
    return args


def test_real_entry_preparation_epoch_recovery_completion_and_wrong_identity(synthetic_run):
    args = synthetic_run
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        args.stop_after_epoch = 1
        first = prepare("satellite", args)
        output = execute(first, args)
        identity = first["identity"]
        assert not is_complete(output, identity)
        before = (output / "latest.pt").read_bytes()
        wrong = copy.deepcopy(identity)
        wrong["feature_statistics_sha256"] = "other"
        first["identity"] = wrong
        args.resume_run = output
        with pytest.raises(ValueError, match="identity mismatch"):
            execute(first, args)
        assert (output / "latest.pt").read_bytes() == before
        args.stop_after_epoch = None
        second = prepare("satellite", args)
        assert second["identity"] == identity
        execute(second, args)
        assert is_complete(output, identity)
        completed = (output / "complete.json").read_bytes()
        execute(prepare("satellite", args), args)
        assert (output / "complete.json").read_bytes() == completed
        metrics = json.loads((output / "metrics.json").read_text())
        assert set(metrics) == {"validation_development", "validation_holdout"}
        assert set(metrics["validation_development"]) == {"none", "zero", "shuffle"}
        for partition in metrics.values():
            assert len({json_hash(v["comparison"]) for v in partition.values()}) == 1
            assert partition["none"]["daylight_primary_mae"] is not None
        assert identity["scope"] == "smoke"
        assert not identity["test_unlocked"]
        args.unlock_test = True
        with pytest.raises(ValueError, match="cannot unlock"):
            prepare("satellite", args)
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize("field", ["config_hash", "effective_config_hash", "local_data_hash",
                                   "split_manifest_hash", "satellite_statistics_sha256",
                                   "source_hash", "environment_hash", "selection_metric"])
def test_identity_fields_all_prevent_completion_reuse(tmp_path, field):
    from cloud2watt.run_state import REQUIRED_ARTIFACTS, mark_complete
    for name in REQUIRED_ARTIFACTS:
        (tmp_path / name).write_bytes(b"fixture")
    identity = {"protocol": PROTOCOL, field: "original"}
    mark_complete(tmp_path, identity)
    assert not is_complete(tmp_path, identity | {field: "changed"})
