"""Train and evaluate the leakage-safe lightweight CNN late-fusion model."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from cloud2watt.data.satellite_loader import (
    ChunkBucketSampler,
    SatelliteForecastDataset,
    fit_satellite_statistics,
)
from cloud2watt.evaluation.splits import build_split_manifest
from cloud2watt.models import SatelliteLateFusion
from cloud2watt.satellite_training import predict_satellite, run_satellite_epoch
from cloud2watt.training import FeatureStatistics, metrics_by_horizon, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/paired-30d-v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/cnn_late_fusion.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--mode", choices=tuple(sorted(SatelliteLateFusion.MODES)))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--unlock-test", action="store_true")
    return parser.parse_args()


def make_loader(dataset: SatelliteForecastDataset, rows: pd.DataFrame, config: dict,
                *, training: bool, seed: int) -> DataLoader:
    workers = int(config["num_workers"])
    kwargs: dict[str, object] = {
        "batch_size": int(config["batch_size"]),
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if training:
        kwargs["sampler"] = ChunkBucketSampler(
            rows, time_chunk=int(config["time_chunk"]), shuffle_buckets=True, seed=seed
        )
    if workers:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(config["prefetch_factor"])
    return DataLoader(dataset, **kwargs)


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = args.seed if args.seed is not None else int(config["random_seeds"][0])
    seed_everything(seed)
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset_version"] != config["data_version"]:
        raise SystemExit("data version does not match CNN config")
    samples = pd.read_parquet(args.data / "sample_index.parquet")
    power = pd.read_parquet(args.data / "power_15min.parquet")
    sites = pd.read_parquet(args.data / "sites.parquet")
    assigned, split_manifest = build_split_manifest(
        samples, sites, data_manifest_hash=manifest["manifest_content_sha256"],
        seed=42, embargo_hours=int(config["embargo_hours"]),
    )
    train = assigned.loc[
        assigned["temporal_split"].eq("train") & assigned["site_split"].eq("development")
    ]
    if args.max_train_samples is not None:
        train = train.iloc[:args.max_train_samples]
    validation = assigned.loc[
        assigned["temporal_split"].eq("validation")
        & assigned["site_split"].eq("development")
    ]
    if args.max_validation_samples is not None:
        validation = validation.iloc[:args.max_validation_samples]
    feature_statistics = FeatureStatistics.fit(train, power, sites)
    satellite_path = args.data / "satellite_frames.zarr"
    satellite_statistics = fit_satellite_statistics(
        train, satellite_path,
        maximum_frames=int(config["satellite_statistics"]["maximum_frames"]),
    )

    def dataset(rows: pd.DataFrame, *, augment: bool = False) -> SatelliteForecastDataset:
        return SatelliteForecastDataset(
            rows, power, sites, feature_statistics, satellite_statistics, satellite_path,
            augment=augment, jitter_padding=4 if augment else 0, seed=seed,
        )

    loader_config = config["training"]
    train_loader = make_loader(dataset(train, augment=True), train, loader_config,
                               training=True, seed=seed)
    validation_loader = make_loader(dataset(validation), validation, loader_config,
                                    training=False, seed=seed)
    mode = args.mode or str(config["model"]["mode"])
    model_args = {key: value for key, value in config["model"].items() if key != "mode"}
    model = SatelliteLateFusion(mode=mode, **model_args)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters > 10_000_000:
        raise ValueError("late-fusion model exceeds the 10M parameter budget")
    if mode == "full" and parameters < 5_000_000:
        raise ValueError("full late-fusion model is below the frozen 5M parameter budget")
    device_name = str(loader_config["device"])
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(loader_config["learning_rate"]),
        weight_decay=float(loader_config["weight_decay"]),
    )
    max_epochs = args.max_epochs or int(loader_config["max_epochs"])
    use_amp = bool(loader_config["amp"]) and device.type == "cuda"
    identity = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config_hash": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "data_manifest_hash": manifest["manifest_content_sha256"],
        "split_manifest_hash": split_manifest["split_manifest_sha256"],
        "feature_statistics_sha256": feature_statistics.sha256,
        "satellite_statistics_sha256": satellite_statistics.sha256,
        "mode": mode, "parameters": parameters, "seed": seed,
        "max_epochs": max_epochs, "max_train_samples": args.max_train_samples,
        "max_validation_samples": args.max_validation_samples,
        "test_unlocked": args.unlock_test,
    }
    run_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    output = args.resume_run or args.output_root / f"{config['run_name']}-{mode}-s{seed}-{run_hash}"
    if args.resume_run is None:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise ValueError(f"resume run does not exist: {output}")
    checkpoint = output / "best.pt"
    latest_checkpoint = output / "latest.pt"
    best, stale, history, first_epoch = np.inf, 0, [], 1
    if args.resume_run is not None:
        resume_checkpoint = latest_checkpoint if latest_checkpoint.exists() else checkpoint
        payload = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        previous = payload["metadata"]
        immutable = (
            "config_hash", "data_manifest_hash", "split_manifest_hash", "mode", "parameters",
            "seed", "max_epochs", "max_train_samples", "max_validation_samples", "test_unlocked",
        )
        mismatched = [key for key in immutable if previous.get(key) != identity.get(key)]
        if mismatched:
            raise ValueError(f"resume metadata mismatch: {', '.join(mismatched)}")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        best = float(payload.get("best_validation_loss", payload["validation_loss"]))
        stale = int(payload.get("stale_epochs", 0))
        history = list(payload.get("history", []))
        first_epoch = int(payload["epoch"]) + 1
        identity = previous | {"resumed_with_git_commit": identity["git_commit"]}
    for epoch in range(first_epoch, max_epochs + 1):
        train_loss = run_satellite_epoch(
            model, train_loader, device=device, optimizer=optimizer, use_amp=use_amp,
            accumulation_steps=int(loader_config["accumulation_steps"]),
        )
        validation_loss = run_satellite_epoch(model, validation_loader, device=device)
        history.append({"epoch": epoch, "train_mae": train_loss,
                        "validation_mae": validation_loss})
        improved = validation_loss < best
        if improved:
            best, stale = validation_loss, 0
        else:
            stale += 1
        state = {
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "epoch": epoch, "validation_loss": validation_loss,
            "best_validation_loss": best, "stale_epochs": stale, "history": history,
            "feature_statistics": asdict(feature_statistics),
            "satellite_statistics": asdict(satellite_statistics), "metadata": identity,
        }
        torch.save(state, latest_checkpoint)
        if improved:
            torch.save(state, checkpoint)
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        print(
            f"epoch={epoch} train_mae={train_loss:.6f} "
            f"validation_mae={validation_loss:.6f} best={best:.6f}",
            flush=True,
        )
        if stale >= int(loader_config["patience"]):
            break
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    partitions = {
        "validation_development": validation,
        "validation_holdout": assigned.loc[
            assigned["temporal_split"].eq("validation")
            & assigned["site_split"].eq("holdout")
        ],
    }
    if args.max_validation_samples is not None:
        partitions = {
            name: rows.iloc[:args.max_validation_samples] for name, rows in partitions.items()
        }
    if args.unlock_test:
        for site_split in ("development", "holdout"):
            partitions[f"test_{site_split}"] = assigned.loc[
                assigned["temporal_split"].eq("test")
                & assigned["site_split"].eq(site_split)
            ]
    metrics, prediction_frames = {}, []
    perturbations = ("none", "zero", "shuffle") if mode == "full" else ("none",)
    for partition, rows in partitions.items():
        loader = make_loader(dataset(rows), rows, loader_config, training=False, seed=seed)
        partition_metrics = {}
        frame = rows.loc[:, ["site_id", "issue_time_utc"]].reset_index(drop=True)
        frame["partition"] = partition
        for perturbation in perturbations:
            prediction, target, mask = predict_satellite(
                model, loader, device, perturbation=perturbation
            )
            name = mode if perturbation == "none" else f"{mode}_satellite_{perturbation}"
            partition_metrics[name] = metrics_by_horizon(prediction, target, mask)
            frame[name] = list(prediction)
        frame["observed"], frame["target_mask"] = list(target), list(mask)
        metrics[partition], prediction_frames = partition_metrics, prediction_frames + [frame]
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        output / "predictions.parquet", index=False
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "provenance.json").write_text(
        json.dumps(identity | {"device": str(device), "amp_used": use_amp},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"CNN late-fusion run written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
