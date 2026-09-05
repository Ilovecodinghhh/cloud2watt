"""Run the leakage-safe Optical Flow ridge baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cloud2watt.evaluation.splits import build_split_manifest
from cloud2watt.optical_flow import MaskedRidge, cached_flow_feature_matrix
from cloud2watt.training import (
    FeatureStatistics,
    PowerForecastDataset,
    metrics_by_horizon,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/paired-30d-v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/optical_flow.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--unlock-test", action="store_true")
    return parser.parse_args()


def tabular_arrays(
    samples: pd.DataFrame,
    power: pd.DataFrame,
    sites: pd.DataFrame,
    statistics: FeatureStatistics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = PowerForecastDataset(samples, power, sites, statistics)
    features, targets, masks = [], [], []
    for record in dataset.records:
        features.append(np.concatenate((record["power_history"], record["solar_future"].ravel(),
                                        record["site"])))
        targets.append(record["target"])
        masks.append(record["target_mask"])
    return np.asarray(features), np.asarray(targets), np.asarray(masks, dtype=bool)


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["dataset_version"] != config["data_version"]:
        raise SystemExit("data version does not match Optical Flow config")
    samples = pd.read_parquet(args.data / "sample_index.parquet")
    power = pd.read_parquet(args.data / "power_15min.parquet")
    sites = pd.read_parquet(args.data / "sites.parquet")
    assigned, split_manifest = build_split_manifest(
        samples, sites, data_manifest_hash=manifest["manifest_content_sha256"],
        seed=int(config["random_seed"]), embargo_hours=int(config["embargo_hours"]),
    )
    train = assigned.loc[
        assigned["temporal_split"].eq("train") & assigned["site_split"].eq("development")
    ]
    validation = assigned.loc[
        assigned["temporal_split"].eq("validation")
        & assigned["site_split"].eq("development")
    ]
    statistics = FeatureStatistics.fit(train, power, sites)
    train_tabular, train_target, train_mask = tabular_arrays(train, power, sites, statistics)
    validation_tabular, validation_target, validation_mask = tabular_arrays(
        validation, power, sites, statistics
    )
    satellite_path = str(args.data / "satellite_frames.zarr")
    flow_config = config["satellite"]
    cache_dir = args.output_root / "feature_cache"
    train_flow, train_flow_hash, train_cache_hit = cached_flow_feature_matrix(
        train, satellite_path, cache_dir, ir_channel=int(flow_config["ir_channel"]),
        downsample=int(flow_config["downsample"]),
        time_chunk=int(config["loader"]["time_chunk"]),
    )
    validation_flow, validation_flow_hash, validation_cache_hit = cached_flow_feature_matrix(
        validation, satellite_path, cache_dir, ir_channel=int(flow_config["ir_channel"]),
        downsample=int(flow_config["downsample"]),
        time_chunk=int(config["loader"]["time_chunk"]),
    )
    candidates = []
    for alpha in config["ridge"]["alpha_candidates"]:
        model = MaskedRidge(float(alpha)).fit(
            np.column_stack((train_tabular, train_flow)), train_target, train_mask
        )
        forecast = model.predict(np.column_stack((validation_tabular, validation_flow)))
        score = float(np.mean([row["mae"] for row in
                               metrics_by_horizon(forecast, validation_target, validation_mask)]))
        candidates.append((score, float(alpha), model))
    _, selected_alpha, flow_model = min(candidates, key=lambda value: value[0])
    tabular_model = MaskedRidge(selected_alpha).fit(train_tabular, train_target, train_mask)
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
    results: dict[str, object] = {}
    predictions = []
    for partition, rows in partitions.items():
        if partition == "validation_development":
            tabular, target, mask, flow = (
                validation_tabular, validation_target, validation_mask, validation_flow
            )
        else:
            tabular, target, mask = tabular_arrays(rows, power, sites, statistics)
            flow, _partition_flow_hash, _partition_cache_hit = cached_flow_feature_matrix(
                rows, satellite_path, cache_dir, ir_channel=int(flow_config["ir_channel"]),
                downsample=int(flow_config["downsample"]),
                time_chunk=int(config["loader"]["time_chunk"]),
            )
        tabular_forecast = tabular_model.predict(tabular)
        flow_forecast = flow_model.predict(np.column_stack((tabular, flow)))
        results[partition] = {
            "power_solar_ridge": metrics_by_horizon(tabular_forecast, target, mask),
            "optical_flow_ridge": metrics_by_horizon(flow_forecast, target, mask),
        }
        frame = rows.loc[:, ["site_id", "issue_time_utc"]].reset_index(drop=True)
        frame["partition"] = partition
        frame["observed"], frame["target_mask"] = list(target), list(mask)
        frame["power_solar_ridge"], frame["optical_flow_ridge"] = (
            list(tabular_forecast), list(flow_forecast)
        )
        predictions.append(frame)
    identity = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config_hash": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "data_manifest_hash": manifest["manifest_content_sha256"],
        "split_manifest_hash": split_manifest["split_manifest_sha256"],
        "feature_statistics_sha256": statistics.sha256,
        "selected_alpha": selected_alpha,
        "flow_cache": {
            "train_hash": train_flow_hash,
            "validation_hash": validation_flow_hash,
            "train_cache_hit": train_cache_hit,
            "validation_cache_hit": validation_cache_hit,
        },
        "test_unlocked": args.unlock_test,
    }
    run_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    output = args.output_root / f"{config['run_name']}-{run_hash}"
    output.mkdir(parents=True, exist_ok=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        output / "predictions.parquet", index=False
    )
    (output / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "provenance.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Optical Flow run written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
