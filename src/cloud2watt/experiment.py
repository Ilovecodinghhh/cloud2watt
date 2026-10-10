"""Shared V2 preparation and execution for power and satellite experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from cloud2watt.data.development import read_power_for_samples
from cloud2watt.data.satellite_loader import (
    ChunkBucketSampler,
    SatelliteForecastDataset,
    fit_satellite_statistics,
)
from cloud2watt.epoch import evaluate, fit_epochs
from cloud2watt.evaluation.forecast import (
    PROTOCOL,
    SELECTION_METRIC,
    ForecastError,
    comparison_identity,
    require_primary,
    score_forecast,
)
from cloud2watt.evaluation.splits import build_split_manifest
from cloud2watt.models import SatelliteLateFusion, build_power_model
from cloud2watt.run_state import (
    atomic_json,
    atomic_path,
    environment_snapshot,
    file_hash,
    is_complete,
    json_hash,
    mark_complete,
    run_lock,
    source_snapshot,
)
from cloud2watt.training import FeatureStatistics, PowerForecastDataset, seed_everything

ROOT = Path(__file__).resolve().parents[2]


def parser_for(kind):
    parser = argparse.ArgumentParser(description=f"V2 {kind} development training")
    parser.add_argument("--data", type=Path, default=Path("data/processed/paired-30d-v1"))
    config = "cnn_late_fusion" if kind == "satellite" else "power_models"
    parser.add_argument("--config", type=Path, default=Path(f"configs/{config}.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--mode", choices=sorted(SatelliteLateFusion.MODES))
    parser.add_argument("--model", choices=("mlp", "tcn"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int,
                        help="Graceful epoch-boundary pause; does not change the epoch budget")
    parser.add_argument("--unlock-test", action="store_true")
    return parser


def _limited(rows, power, maximum):
    if maximum is None:
        return rows
    if maximum < 1:
        raise ValueError("sample limits must be positive")
    # Capped acceptance runs need a defined primary score; these are smoke runs.
    indices = np.stack(rows.target_row_indices).astype(int)
    mask = np.stack(rows.target_mask).astype(bool) & (indices >= 0)
    elevation = power.solar_elevation_deg.to_numpy()[np.maximum(indices, 0)]
    rows = rows.loc[(mask[:, [1, 2, 3]] & (elevation[:, [1, 2, 3]] > 5)).all(axis=1)]
    if rows.empty:
        raise ValueError("no daylight samples available for smoke cap")
    return rows.iloc[np.linspace(0, len(rows) - 1, min(maximum, len(rows)), dtype=int)]


def prepare(kind, args):
    if args.unlock_test:
        raise ValueError("S1 development entrypoint cannot unlock test; use frozen S8 evaluation")
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes.decode("utf-8"))
    if config.get("evaluation_protocol") != PROTOCOL:
        raise ValueError(f"config must select {PROTOCOL}; legacy runs are incompatible")
    if config.get("selection", {}).get("metric") != SELECTION_METRIC:
        raise ValueError("config selection metric must be daylight_primary_mae")
    seed = args.seed if args.seed is not None else int(config["random_seeds"][0])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(seed)
    torch.use_deterministic_algorithms(True)
    training = config["training"].copy()
    if args.max_epochs is not None:
        training["max_epochs"] = args.max_epochs
    if min(int(training[k]) for k in ("max_epochs", "batch_size", "patience")) < 1:
        raise ValueError("epochs, batch_size and patience must be positive")
    if not 0 <= int(training.get("min_epochs", 0)) <= int(training["max_epochs"]):
        raise ValueError("min_epochs must be between zero and max_epochs")
    if not 0 <= float(training.get("primary_loss_weight", 0)) < 1:
        raise ValueError("primary_loss_weight must be in [0, 1)")
    augmentation = config.get("augmentation", {"enabled": True, "jitter_padding": 4})
    if int(augmentation.get("jitter_padding", 4)) < 0:
        raise ValueError("jitter_padding must be nonnegative")
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        raise ValueError("stop-after-epoch must be positive")
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset_version"] != config["data_version"]:
        raise ValueError("data version mismatch")
    samples = pd.read_parquet(args.data / "sample_index.parquet")
    sites = pd.read_parquet(args.data / "sites.parquet")
    assigned, split = build_split_manifest(samples, sites,
        data_manifest_hash=manifest["manifest_content_sha256"], seed=42,
        embargo_hours=int(config["embargo_hours"]))
    partitions = {
        "train": assigned.loc[(assigned.temporal_split == "train")
                              & (assigned.site_split == "development")],
        "validation": assigned.loc[(assigned.temporal_split == "validation")
                                   & (assigned.site_split == "development")],
        "holdout": assigned.loc[(assigned.temporal_split == "validation")
                                & (assigned.site_split == "holdout")],
    }
    power = read_power_for_samples(args.data / "power_15min.parquet",
                                  pd.concat(partitions.values(), ignore_index=True))
    for key, rows in partitions.items():
        cap = args.max_train_samples if key == "train" else args.max_validation_samples
        partitions[key] = _limited(rows, power, cap).reset_index(drop=True)
        if partitions[key].empty:
            raise ValueError(f"empty {key} partition")
    diagnostic_cap = int(config.get("diagnostics", {}).get("train_eval_samples", 0))
    if diagnostic_cap < 0:
        raise ValueError("train_eval_samples must be nonnegative")
    if diagnostic_cap:
        partitions["train_eval"] = _limited(partitions["train"], power,
                                           diagnostic_cap).reset_index(drop=True)
    stats = FeatureStatistics.fit(partitions["train"], power, sites)
    satellite_stats = fit_satellite_statistics(partitions["train"],
        args.data / "satellite_frames.zarr",
        maximum_frames=int(config["satellite_statistics"]["maximum_frames"])) \
        if kind == "satellite" else None
    datasets, comparison = {}, {}
    for key, rows in partitions.items():
        base = PowerForecastDataset(rows, power, sites, stats)
        y = np.stack([r["target"] for r in base.records])
        mask = np.stack([r["target_mask"] for r in base.records])
        elevation = np.stack([r["target_solar_elevation_deg"] for r in base.records])
        comparison[key] = comparison_identity(rows, y, mask, elevation)
        if key == "validation":
            require_primary(score_forecast(y, y, mask, elevation, rows, groups=False))
        datasets[key] = SatelliteForecastDataset(rows, power, sites, stats, satellite_stats,
            args.data / "satellite_frames.zarr",
            augment=key == "train" and bool(augmentation.get("enabled", True)),
            jitter_padding=int(augmentation.get("jitter_padding", 4)),
            seed=seed, spatial_pool=int(config.get("spatial_pool", 1)),
            load_satellite=(args.mode or config["model"]["mode"]) != "power_solar"
            ) if kind == "satellite" else base
    model_args = {k: v for k, v in config["model"].items() if k not in ("mode", "name")}
    if kind == "satellite":
        mode = args.mode or config["model"]["mode"]
        model = SatelliteLateFusion(mode=mode, **model_args)
    else:
        mode = args.model or config["model"]["name"]
        if mode == "mlp":
            model_args = {"hidden_size": model_args.get("channels", 128),
                          "dropout": model_args.get("dropout", 0.1)}
        model = build_power_model(mode, **model_args)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters > (10_000_000 if kind == "satellite" else 999_999):
        raise ValueError("model exceeds parameter budget")
    device_name = str(training.get("device", "auto"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model.to(device)
    loaders = {}
    for key, dataset in datasets.items():
        kwargs = {"batch_size": int(training["batch_size"]),
                  "num_workers": int(training.get("num_workers", 0)),
                  "generator": torch.Generator().manual_seed(seed + len(loaders)),
                  "pin_memory": device.type == "cuda"}
        if key == "train" and kind == "satellite":
            kwargs["sampler"] = ChunkBucketSampler(partitions[key],
                time_chunk=int(training["time_chunk"]), shuffle_buckets=True, seed=seed)
        else:
            kwargs["shuffle"] = key == "train"
        # Restart workers every epoch. Jitter is stateless by seed/sample index.
        if kwargs["num_workers"]:
            kwargs["prefetch_factor"] = int(training.get("prefetch_factor", 2))
        loaders[key] = DataLoader(dataset, **kwargs)
    env, source = environment_snapshot(), source_snapshot(ROOT)
    local_data = {name: file_hash(args.data / name) for name in
                  ("manifest.json", "sample_index.parquet", "power_15min.parquet", "sites.parquet")}
    if satellite_stats is not None:
        zarr_path = args.data / "satellite_frames.zarr"
        local_data["satellite_file_inventory"] = json_hash([
            (p.relative_to(zarr_path).as_posix(), p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(zarr_path.rglob("*")) if p.is_file()])
    effective = config | {"training": training, "model": model_args | {"mode": mode}, "seed": seed}
    identity = {"protocol": PROTOCOL, "selection_metric": SELECTION_METRIC,
                "kind": kind, "mode": mode, "seed": seed, "parameters": parameters,
                "config_hash": hashlib.sha256(config_bytes).hexdigest(),
                "effective_config_hash": json_hash(effective),
                "training_config_hash": json_hash(training),
                "model_config_hash": json_hash(model_args | {"mode": mode}),
                "data_manifest_hash": manifest["manifest_content_sha256"],
                "local_data_hash": json_hash(local_data),
                "split_manifest_hash": split["split_manifest_sha256"],
                "comparison": comparison, "feature_statistics_sha256": stats.sha256,
                "satellite_statistics_sha256": satellite_stats.sha256 if satellite_stats else None,
                "source_hash": json_hash(source), "environment_hash": json_hash(env),
                "device": str(device), "amp_used": bool(training.get("amp"))
                and device.type == "cuda", "max_epochs": int(training["max_epochs"]),
                "max_train_samples": args.max_train_samples,
                "max_validation_samples": args.max_validation_samples, "test_unlocked": False,
                "data_role": "legacy_development", "scope": "smoke" if
                args.max_train_samples is not None or args.max_validation_samples is not None
                else "development"}
    return {"identity": identity, "model": model, "loaders": loaders, "device": device,
            "config": effective, "config_bytes": config_bytes, "split": split,
            "environment": env, "source": source,
            "statistics": {"feature": asdict(stats), "satellite": asdict(satellite_stats)
                           if satellite_stats else None}, "run_name": config["run_name"]}


def execute(prepared, args):
    identity = prepared["identity"]
    expected = args.output_root / (f"{prepared['run_name']}-{identity['mode']}-"
                                   f"s{identity['seed']}-{json_hash(identity)[:12]}")
    output = args.resume_run or expected
    if args.resume_run is not None and not output.is_dir():
        raise ValueError("resume directory does not exist")
    if output.exists() and not (args.resume_run or args.auto_resume):
        raise ValueError("run exists; use --resume-run or --auto-resume")
    with run_lock(output):
        if is_complete(output, identity):
            print(f"verified completed run: {output}", flush=True)
            return output
        if (output / "complete.json").exists():
            raise ValueError("completed run identity/artifacts differ; refusing overwrite")
        resume = (output / "latest.pt").exists()
        if args.resume_run is not None and not resume:
            raise ValueError("resume requires a complete V2 latest.pt epoch")
        if resume:
            from cloud2watt.run_state import load_epoch
            load_epoch(output / "latest.pt", identity)
        elif (output / "provenance.json").exists():
            old = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
            if old.get("identity") != identity:
                raise ValueError("existing run identity mismatch")
        with atomic_path(output / "config.yaml") as temporary:
            temporary.write_bytes(prepared["config_bytes"])
        # Archive dirty/new source too, not just HEAD or source hashes.
        with atomic_path(output / "source.zip") as temporary:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, expected_hash in prepared["source"].items():
                    content = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
                    if hashlib.sha256(content).hexdigest() != expected_hash:
                        raise ValueError("source changed after preparation; rebuild run identity")
                    archive.writestr(name, content)
        for name, key in (("config.json", "config"), ("environment.json", "environment"),
                          ("statistics.json", "statistics"), ("source.json", "source"),
                          ("split_manifest.json", "split")):
            atomic_json(output / name, prepared[key])
        atomic_json(output / "provenance.json", {"identity": identity,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                   cwd=ROOT, text=True).strip()})
        model, training = prepared["model"], prepared["config"]["training"]
        optimizer_type = torch.optim.AdamW if identity["kind"] == "satellite" else torch.optim.Adam
        optimizer = optimizer_type(model.parameters(), lr=float(training["learning_rate"]),
                                   weight_decay=float(training.get("weight_decay", 0)))
        try:
            done = fit_epochs(model, optimizer, prepared["loaders"], device=prepared["device"],
                identity=identity, output=output, max_epochs=identity["max_epochs"],
                patience=int(training["patience"]), satellite=identity["kind"] == "satellite",
                accumulation_steps=int(training.get("accumulation_steps", 1)),
                use_amp=identity["amp_used"], resume=resume, stop_after_epoch=args.stop_after_epoch,
                min_epochs=int(training.get("min_epochs", 0)),
                primary_loss_weight=float(training.get("primary_loss_weight", 0)))
            if not done:
                print(f"paused at committed epoch: {output}", flush=True)
                return output
            metrics, frames = {}, []
            for key, partition in (("validation", "validation_development"),
                                   ("holdout", "validation_holdout")):
                metrics[partition] = {}
                perturbations = ("none", "zero", "shuffle") if identity["mode"] == "full" \
                    else ("none",)
                for perturbation in perturbations:
                    report, frame = evaluate(model, prepared["loaders"][key], prepared["device"],
                        satellite=identity["kind"] == "satellite", perturbation=perturbation)
                    if report["comparison"] != identity["comparison"][key]:
                        raise ValueError("evaluation target population changed")
                    metrics[partition][perturbation] = report
                    frames.append(frame.assign(partition=partition, perturbation=perturbation))
            atomic_json(output / "metrics.json", metrics)
            with atomic_path(output / "predictions.parquet") as temporary:
                pd.concat(frames, ignore_index=True).to_parquet(temporary, index=False)
            mark_complete(output, identity)
        except Exception as exc:
            report = exc.report if isinstance(exc, ForecastError) else {"error": str(exc)}
            atomic_json(output / "failure.json", {"identity": identity, "failure": report})
            raise
    print(f"V2 development run written to {output}", flush=True)
    return output


def main(kind):
    args = parser_for(kind).parse_args()
    execute(prepare(kind, args), args)
    return 0
