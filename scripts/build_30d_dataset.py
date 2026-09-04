"""Build the immutable 20-site, 30-day Cloud2Watt dataset with full QC."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import tracemalloc
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

from cloud2watt.data.paired import (
    FORECAST_MINUTES,
    HISTORY_MINUTES,
    build_sample_index,
    hash_sample_keys,
    paired_time_grid,
    partition_satellite_times,
    prepare_paired_power,
    select_compact_sites,
    validate_sample_index,
    write_paired_metadata,
    write_satellite_frames,
)
from cloud2watt.data.quality import (
    QualityFlag,
    flag_context_quality,
    missing_site_metadata,
    quality_breakdown,
)
from cloud2watt.data.seviri import SeviriStore
from cloud2watt.data.solar import build_solar_features
from scripts.build_paired_mini_dataset import (
    DEFAULT_STORE,
    apply_bad_data,
    benchmark_reads,
    write_projection_preview,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power", type=Path, nargs="+", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--bad-data", type=Path, required=True)
    parser.add_argument("--pv-revision", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/data/paired-30d-v1.yaml"))
    parser.add_argument("--store-url", default=DEFAULT_STORE)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/cache/seviri-2020"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/paired-30d-v1"))
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_power(paths: list[Path], start: datetime, end: datetime) -> pd.DataFrame:
    frames = []
    read_start = pd.Timestamp(start + timedelta(minutes=min(HISTORY_MINUTES) - 15))
    read_end = pd.Timestamp(end + timedelta(minutes=max(FORECAST_MINUTES)))
    for path in paths:
        frames.append(
            pd.read_parquet(
                path,
                filters=[("datetime_GMT", ">", read_start), ("datetime_GMT", "<=", read_end)],
            )
        )
    return pd.concat(frames, ignore_index=True)


def annual_storage_estimates(total_bytes: int, days: int, sites: int) -> dict[str, int]:
    return {
        f"{target_sites}_sites_one_year_bytes": round(
            total_bytes * 365 / days * target_sites / sites
        )
        for target_sites in (100, 300)
    }


def validate_new_version_path(output: Path, dataset_version: str) -> Path:
    """Return a new immutable version path after validating its name and absence."""
    if output.name != dataset_version:
        raise ValueError("output directory name must equal config dataset_version")
    if output.exists():
        raise FileExistsError(f"immutable dataset version already exists: {output}")
    return output


def main() -> int:
    args = parse_args()
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    try:
        output = validate_new_version_path(args.output, str(config.get("dataset_version")))
    except (ValueError, FileExistsError) as error:
        raise SystemExit(str(error)) from error
    issue_start = datetime.fromisoformat(
        config["issue_start_utc"].replace("Z", "+00:00")
    ).astimezone(UTC)
    issue_end = issue_start + timedelta(days=int(config["days"]))
    started = time.perf_counter()
    tracemalloc.start()

    power_source = load_power(args.power, issue_start, issue_end)
    metadata = pd.read_csv(args.metadata)
    bad_data = pd.read_csv(args.bad_data)
    sites = select_compact_sites(
        power_source,
        metadata,
        start_utc=issue_start,
        end_utc=issue_end,
        site_count=int(config["site_count"]),
    )
    missing_metadata = missing_site_metadata(sites)
    if missing_metadata.any():
        missing_ids = sites.loc[missing_metadata, "ss_id"].astype(str).tolist()
        raise ValueError(f"selected sites have incomplete metadata: {missing_ids}")

    power, _ = prepare_paired_power(
        power_source,
        sites,
        issue_start_utc=issue_start,
        issue_end_utc=issue_end,
        value_semantics=config["power_value_semantics"],
    )
    solar = build_solar_features(
        pd.DatetimeIndex(power["datetime_GMT"].drop_duplicates().sort_values()), sites
    )
    power = power.merge(
        solar,
        on=["ss_id", "datetime_GMT"],
        how="left",
        validate="one_to_one",
    )
    quality = config["quality"]
    power, _ = flag_context_quality(
        power,
        night_elevation_degrees=float(quality["night_elevation_degrees"]),
        night_nonzero_fraction=float(quality["night_nonzero_fraction"]),
        daytime_elevation_degrees=float(quality["daytime_elevation_degrees"]),
        stuck_zero_bins=int(quality["stuck_zero_bins"]),
    )
    power, _ = apply_bad_data(power, bad_data)

    client = SeviriStore(args.store_url, cache_dir=args.cache_dir)
    issue_times, requested_satellite_times = paired_time_grid(issue_start, issue_end)
    satellite_times, source_indices, missing_satellite_times = partition_satellite_times(
        requested_satellite_times, client.timestamps()
    )
    site_pixels = client.map_sites(sites)
    sites["satellite_x_index"] = [site.x_index for site in site_pixels]
    sites["satellite_y_index"] = [site.y_index for site in site_pixels]
    output.mkdir(parents=True, exist_ok=False)
    satellite_report = write_satellite_frames(
        output / "satellite_frames.zarr",
        client,
        site_pixels,
        requested_times=satellite_times,
        source_time_indices=source_indices,
        channel_names=list(config["channels"]),
        height=int(config["crop_shape"][0]),
        width=int(config["crop_shape"][1]),
        quality_output_path=output / "satellite_quality.parquet",
        maximum_invalid_fraction=float(quality["maximum_satellite_invalid_fraction"]),
    )
    satellite_report["missing_critical_frames"] = len(missing_satellite_times) * len(sites)
    frame_quality_path = output / "satellite_quality.parquet"
    frame_quality = pd.read_parquet(frame_quality_path)
    if len(missing_satellite_times):
        missing_rows = pd.DataFrame(
            [
                {
                    "ss_id": str(site_id),
                    "datetime_GMT": timestamp,
                    "invalid_fraction": 1.0,
                    "quality_flags": int(QualityFlag.SATELLITE_MISSING),
                    "is_valid": False,
                }
                for site_id in sites["ss_id"]
                for timestamp in missing_satellite_times
            ]
        )
        frame_quality = pd.concat([frame_quality, missing_rows], ignore_index=True)
        frame_quality.to_parquet(frame_quality_path, index=False)
    invalid_satellite_keys = {
        (str(row.ss_id), pd.Timestamp(row.datetime_GMT))
        for row in frame_quality.loc[~frame_quality["is_valid"]].itertuples(index=False)
    }
    sample_index, sample_stats = build_sample_index(
        power,
        sites,
        issue_times=issue_times,
        satellite_times=satellite_times,
        invalid_satellite_keys=invalid_satellite_keys,
    )
    validation = validate_sample_index(sample_index, power, satellite_times)
    config_sha = sha256_bytes(config_bytes)
    manifest = {
        "schema_version": 2,
        "dataset_version": config["dataset_version"],
        "source": {
            "pv": "openclimatefix/uk_pv",
            "pv_revision": args.pv_revision,
            "satellite": args.store_url,
            "satellite_metadata_sha256": client.metadata_sha256,
        },
        "config_sha256": config_sha,
        "issue_start_exclusive_utc": issue_start.isoformat().replace("+00:00", "Z"),
        "issue_end_inclusive_utc": issue_end.isoformat().replace("+00:00", "Z"),
        "days": int(config["days"]),
        "site_count": len(sites),
        "site_ids": sites["ss_id"].astype(str).tolist(),
        "channels": list(config["channels"]),
        "crop_shape": list(config["crop_shape"]),
        "history_minutes": list(HISTORY_MINUTES),
        "forecast_minutes": list(FORECAST_MINUTES),
        "power_value_semantics": config["power_value_semantics"],
        "issue_times_per_site": len(issue_times),
        "requested_satellite_frame_count_per_site": len(requested_satellite_times),
        "stored_satellite_frame_count_per_site": len(satellite_times),
        "sample_count": len(sample_index),
        "sample_key_sha256": hash_sample_keys(sample_index),
    }
    manifest_payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_content_sha256"] = sha256_bytes(manifest_payload)
    build_report: dict[str, object] = {
        "quality": quality_breakdown(power),
        "satellite_quality": quality_breakdown(frame_quality),
        "site_metadata_missing": int(missing_metadata.sum()),
        "satellite": satellite_report,
        "sample_stats": sample_stats,
        "validation": validation,
    }
    write_paired_metadata(
        output,
        power=power,
        sites=sites,
        sample_index=sample_index,
        manifest=manifest,
        build_report=build_report,
    )
    write_projection_preview(
        output, sites, satellite_times, channel_name=str(config["channels"][0])
    )
    build_report["read_benchmark"] = benchmark_reads(output)
    build_report["cached_source_bytes"] = sum(
        path.stat().st_size for path in args.cache_dir.rglob("*") if path.is_file()
    )
    build_report["total_bytes"] = sum(
        path.stat().st_size for path in output.rglob("*") if path.is_file()
    )
    build_report["annual_storage_estimates"] = annual_storage_estimates(
        int(build_report["total_bytes"]), int(config["days"]), len(sites)
    )
    build_report["total_elapsed_seconds"] = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    build_report["peak_python_traced_memory_bytes"] = peak
    (output / "build_report.json").write_text(
        json.dumps(build_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{config['dataset_version']} written: {len(sample_index)} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
