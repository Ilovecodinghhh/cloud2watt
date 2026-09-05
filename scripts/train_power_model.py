"""Train a leakage-safe power-only MLP or TCN."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from cloud2watt.evaluation.splits import build_split_manifest
from cloud2watt.models import build_power_model
from cloud2watt.training import (
    FeatureStatistics,
    PowerForecastDataset,
    create_loader,
    load_checkpoint,
    metrics_by_horizon,
    predict,
    run_epoch,
    save_checkpoint,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/paired-30d-v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/power_models.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--model", choices=("mlp", "tcn"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--unlock-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = args.seed if args.seed is not None else int(config["random_seeds"][0])
    seed_everything(seed)
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["dataset_version"] != config["data_version"]:
        raise SystemExit("data version does not match training config")
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
    validation = assigned.loc[
        assigned["temporal_split"].eq("validation")
        & assigned["site_split"].eq("development")
    ]
    statistics = FeatureStatistics.fit(train, power, sites)
    training = config["training"]
    train_loader = create_loader(
        PowerForecastDataset(train, power, sites, statistics),
        batch_size=int(training["batch_size"]), shuffle=True, seed=seed,
    )
    validation_loader = create_loader(
        PowerForecastDataset(validation, power, sites, statistics),
        batch_size=int(training["batch_size"]), shuffle=False, seed=seed,
    )
    name = args.model or str(config["model"]["name"])
    model_config = {key: value for key, value in config["model"].items() if key != "name"}
    if name == "mlp":
        model_config = {"hidden_size": int(model_config.get("channels", 128)),
                        "dropout": float(model_config.get("dropout", 0.1))}
    device_setting = str(training["device"])
    if device_setting == "auto":
        device_setting = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_setting)
    model = build_power_model(name, **model_config).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters >= 1_000_000:
        raise ValueError("power-only model must remain below one million parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    max_epochs = args.max_epochs or int(training["max_epochs"])
    identity = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config_hash": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "data_manifest_hash": manifest["manifest_content_sha256"],
        "split_manifest_hash": split_manifest["split_manifest_sha256"],
        "feature_statistics_sha256": statistics.sha256,
        "model": name, "parameters": parameters, "seed": seed,
        "max_epochs": max_epochs,
        "test_unlocked": args.unlock_test,
    }
    run_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    output = args.output_root / f"{config['run_name']}-{name}-s{seed}-{run_hash}"
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "best.pt"
    best, stale, history = np.inf, 0, []
    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, train_loader, device=device, optimizer=optimizer)
        validation_loss = run_epoch(model, validation_loader, device=device)
        history.append({"epoch": epoch, "train_mae": train_loss,
                        "validation_mae": validation_loss})
        if validation_loss < best:
            best, stale = validation_loss, 0
            save_checkpoint(checkpoint, model, optimizer, epoch=epoch,
                            validation_loss=validation_loss, statistics=statistics,
                            metadata=identity)
        else:
            stale += 1
            if stale >= int(training["patience"]):
                break
    load_checkpoint(checkpoint, model)
    partitions = {
        "validation_development": validation,
        "validation_holdout": assigned.loc[
            assigned["temporal_split"].eq("validation")
            & assigned["site_split"].eq("holdout")
        ],
    }
    if args.unlock_test:
        for site_split in ("development", "holdout"):
            partitions[f"test_{site_split}"] = assigned.loc[
                assigned["temporal_split"].eq("test")
                & assigned["site_split"].eq(site_split)
            ]
    results, prediction_frames = {}, []
    for partition, rows in partitions.items():
        loader = create_loader(PowerForecastDataset(rows, power, sites, statistics),
                               batch_size=int(training["batch_size"]), shuffle=False, seed=seed)
        forecast, target, mask = predict(model, loader, device)
        results[partition] = metrics_by_horizon(forecast, target, mask)
        frame = rows.loc[:, ["site_id", "issue_time_utc"]].reset_index(drop=True)
        frame["partition"], frame["prediction"] = partition, list(forecast)
        frame["observed"], frame["target_mask"] = list(target), list(mask)
        prediction_frames.append(frame)
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    pd.concat(prediction_frames).to_parquet(output / "predictions.parquet", index=False)
    (output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (output / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "provenance.json").write_text(
        json.dumps(identity | {"device": str(device)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"power-only run written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
