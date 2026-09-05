"""Benchmark real lazy satellite batches without reading the locked test split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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
from cloud2watt.training import FeatureStatistics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/paired-30d-v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/optical_flow.yaml"))
    parser.add_argument("--batches", type=int, default=1000)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/satellite-loader-benchmark.json")
    )
    args = parser.parse_args()
    if args.batches < 1:
        raise SystemExit("--batches must be positive")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    samples = pd.read_parquet(args.data / "sample_index.parquet")
    power = pd.read_parquet(args.data / "power_15min.parquet")
    sites = pd.read_parquet(args.data / "sites.parquet")
    assigned, split_manifest = build_split_manifest(
        samples, sites, data_manifest_hash=manifest["manifest_content_sha256"], seed=42,
        embargo_hours=int(config["embargo_hours"]),
    )
    train = assigned.loc[
        assigned["temporal_split"].eq("train") & assigned["site_split"].eq("development")
    ]
    satellite_path = args.data / "satellite_frames.zarr"
    statistics_started = time.perf_counter()
    satellite_statistics = fit_satellite_statistics(
        train, satellite_path,
        maximum_frames=int(config["satellite"]["statistics_maximum_frames"]),
    )
    statistics_seconds = time.perf_counter() - statistics_started
    dataset = SatelliteForecastDataset(
        train, power, sites, FeatureStatistics.fit(train, power, sites),
        satellite_statistics, satellite_path,
    )
    loader_config = config["loader"]
    loader = DataLoader(
        dataset, batch_size=int(loader_config["batch_size"]),
        sampler=ChunkBucketSampler(train, time_chunk=int(loader_config["time_chunk"])),
        num_workers=int(loader_config["num_workers"]),
        prefetch_factor=int(loader_config["prefetch_factor"]),
        persistent_workers=int(loader_config["num_workers"]) > 0,
        pin_memory=torch.cuda.is_available(),
    )
    started, samples_read, batches_read = time.perf_counter(), 0, 0
    for batch in loader:
        samples_read += int(batch["satellite"].shape[0])
        batches_read += 1
        if batches_read >= args.batches:
            break
    elapsed = time.perf_counter() - started
    payload = {
        "requested_batches": args.batches, "batches_read": batches_read,
        "samples_read": samples_read, "elapsed_seconds": elapsed,
        "samples_per_second": samples_read / elapsed,
        "statistics_seconds": statistics_seconds,
        "satellite_statistics_sha256": satellite_statistics.sha256,
        "split_manifest_sha256": split_manifest["split_manifest_sha256"],
        "test_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
