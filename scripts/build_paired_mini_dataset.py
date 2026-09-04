"""Build the real 20-site, seven-day paired Cloud2Watt mini dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from cloud2watt.data.mini import mark_source_bad_periods
from cloud2watt.data.paired import (
    FORECAST_MINUTES,
    HISTORY_MINUTES,
    PairedDataset,
    build_sample_index,
    discover_paired_window,
    hash_sample_keys,
    match_satellite_indices,
    paired_time_grid,
    prepare_paired_power,
    select_compact_sites,
    validate_sample_index,
    write_paired_metadata,
    write_satellite_frames,
)
from cloud2watt.data.quality import QualityFlag
from cloud2watt.data.seviri import SeviriStore

DEFAULT_STORE = (
    "https://storage.googleapis.com/public-datasets-eumetsat-solar-forecasting/"
    "satellite/EUMETSAT/SEVIRI_RSS/v4/2020_nonhrv.zarr"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--bad-data", type=Path)
    parser.add_argument("--pv-revision", required=True)
    parser.add_argument("--store-url", default=DEFAULT_STORE)
    parser.add_argument("--issue-start", default="auto", help="UTC date or 'auto'")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--sites", type=int, default=20)
    parser.add_argument("--channels", nargs="+", default=["VIS006", "IR_016", "IR_108"])
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument(
        "--power-value-semantics",
        choices=("interval_energy", "instantaneous_power"),
        required=True,
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/cache/seviri-2020"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/paired-mini-v1")
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_prepare_output(output: Path, overwrite: bool) -> None:
    if not output.exists():
        output.mkdir(parents=True)
        return
    if not overwrite:
        raise SystemExit(f"output already exists: {output}; pass --overwrite to replace it")
    allowed_root = (Path.cwd() / "data" / "processed").resolve()
    resolved = output.resolve()
    if allowed_root not in resolved.parents:
        raise SystemExit("--overwrite is restricted to a child of data/processed")
    shutil.rmtree(output)
    output.mkdir(parents=True)


def load_source_tables(
    args: argparse.Namespace,
    issue_start: datetime | None,
    issue_end: datetime | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    if issue_start is None or issue_end is None:
        power = pd.read_parquet(args.power)
        metadata = pd.read_csv(args.metadata)
        bad_data = pd.read_csv(args.bad_data) if args.bad_data else None
        return power, metadata, bad_data
    read_start = pd.Timestamp(issue_start + timedelta(minutes=min(HISTORY_MINUTES) - 15))
    read_end = pd.Timestamp(issue_end + timedelta(minutes=max(FORECAST_MINUTES)))
    power = pd.read_parquet(
        args.power,
        filters=[("datetime_GMT", ">", read_start), ("datetime_GMT", "<=", read_end)],
    )
    metadata = pd.read_csv(args.metadata)
    bad_data = pd.read_csv(args.bad_data) if args.bad_data else None
    return power, metadata, bad_data


def apply_bad_data(
    power: pd.DataFrame, bad_data: pd.DataFrame | None
) -> tuple[pd.DataFrame, int]:
    if bad_data is None:
        return power, 0
    flagged = mark_source_bad_periods(power, bad_data)
    count = int(
        (
            flagged["quality_flags"].astype("uint16")
            & int(QualityFlag.SOURCE_BAD_PERIOD)
        )
        .ne(0)
        .sum()
    )
    return flagged, count


def benchmark_reads(output: Path, *, count: int = 100) -> dict[str, float]:
    dataset = PairedDataset(output)
    indices = np.linspace(0, len(dataset) - 1, min(count, len(dataset)), dtype=int)
    durations = []
    for index in indices:
        started = time.perf_counter()
        dataset[int(index)]
        durations.append((time.perf_counter() - started) * 1000)
    return {
        "sample_read_median_ms": float(np.median(durations)),
        "sample_read_p95_ms": float(np.percentile(durations, 95)),
    }


def write_projection_preview(
    output: Path,
    sites: pd.DataFrame,
    satellite_times: pd.DatetimeIndex,
    *,
    channel_name: str,
    count: int = 5,
) -> Path:
    """Render several real crops for manual projection-centre inspection."""
    import matplotlib.pyplot as plt

    frames = zarr.open_group(output / "satellite_frames.zarr", mode="r")["frames"]
    noon = satellite_times[0].ceil("D") + pd.Timedelta(hours=12)
    time_index = int(satellite_times.get_indexer([noon], method="nearest")[0])
    preview_count = min(count, len(sites))
    figure, axes = plt.subplots(1, preview_count, figsize=(4 * preview_count, 4))
    axes = np.atleast_1d(axes)
    for site_index, axis in enumerate(axes):
        image = frames[site_index, time_index, 0]
        axis.imshow(image, cmap="gray")
        axis.scatter([image.shape[1] // 2], [image.shape[0] // 2], color="red", marker="+")
        axis.set_title(f"site {sites['ss_id'].iloc[site_index]}\n{channel_name}")
        axis.set_axis_off()
    figure.suptitle(str(satellite_times[time_index]))
    figure.tight_layout()
    path = output / "projection_preview.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def main() -> int:
    args = parse_args()
    issue_start = (
        None
        if args.issue_start == "auto"
        else datetime.fromisoformat(args.issue_start).replace(tzinfo=UTC)
    )
    issue_end = None if issue_start is None else issue_start + timedelta(days=args.days)
    started = time.perf_counter()
    power_source, metadata, bad_data = load_source_tables(args, issue_start, issue_end)
    client = SeviriStore(args.store_url, cache_dir=args.cache_dir)
    available_satellite_times = client.timestamps()
    if issue_start is None or issue_end is None:
        issue_start, issue_end, sites = discover_paired_window(
            power_source,
            metadata,
            available_satellite_times,
            days=args.days,
            site_count=args.sites,
        )
    else:
        sites = select_compact_sites(
            power_source,
            metadata,
            start_utc=issue_start,
            end_utc=issue_end,
            site_count=args.sites,
        )
    power, quality_counts = prepare_paired_power(
        power_source,
        sites,
        issue_start_utc=issue_start,
        issue_end_utc=issue_end,
        value_semantics=args.power_value_semantics,
    )
    power, source_bad_count = apply_bad_data(power, bad_data)
    quality_counts["source_bad_period"] = source_bad_count

    issue_times, satellite_times = paired_time_grid(issue_start, issue_end)
    source_time_indices = match_satellite_indices(
        satellite_times, available_satellite_times
    )
    site_pixels = client.map_sites(sites)
    sites["satellite_x_index"] = [site.x_index for site in site_pixels]
    sites["satellite_y_index"] = [site.y_index for site in site_pixels]
    safe_prepare_output(args.output, args.overwrite)
    satellite_report = write_satellite_frames(
        args.output / "satellite_frames.zarr",
        client,
        site_pixels,
        requested_times=satellite_times,
        source_time_indices=source_time_indices,
        channel_names=args.channels,
        height=args.height,
        width=args.width,
    )
    sample_index, sample_stats = build_sample_index(
        power,
        sites,
        issue_times=issue_times,
        satellite_times=satellite_times,
    )
    validation_stats = validate_sample_index(sample_index, power, satellite_times)
    manifest = {
        "schema_version": 1,
        "source": {
            "pv": "openclimatefix/uk_pv",
            "pv_revision": args.pv_revision,
            "satellite": args.store_url,
            "satellite_metadata_sha256": client.metadata_sha256,
        },
        "issue_start_exclusive_utc": issue_start.isoformat().replace("+00:00", "Z"),
        "issue_end_inclusive_utc": issue_end.isoformat().replace("+00:00", "Z"),
        "days": args.days,
        "site_count": len(sites),
        "site_ids": [str(value) for value in sites["ss_id"]],
        "channels": args.channels,
        "crop_shape": [args.height, args.width],
        "history_minutes": list(HISTORY_MINUTES),
        "forecast_minutes": list(FORECAST_MINUTES),
        "power_value_semantics": args.power_value_semantics,
        "issue_times_per_site": len(issue_times),
        "satellite_frame_count_per_site": len(satellite_times),
        "sample_count": len(sample_index),
        "sample_key_sha256": hash_sample_keys(sample_index),
    }
    build_report: dict[str, object] = {
        "quality_counts": quality_counts,
        "sample_stats": sample_stats,
        "validation": validation_stats,
        "satellite": satellite_report,
    }
    write_paired_metadata(
        args.output,
        power=power,
        sites=sites,
        sample_index=sample_index,
        manifest=manifest,
        build_report=build_report,
    )
    preview_path = write_projection_preview(
        args.output,
        sites,
        satellite_times,
        channel_name=args.channels[0],
    )
    build_report["projection_preview"] = str(preview_path)
    build_report["read_benchmark"] = benchmark_reads(args.output)
    build_report["cached_source_bytes"] = sum(
        path.stat().st_size for path in args.cache_dir.rglob("*") if path.is_file()
    )
    build_report["total_bytes"] = sum(
        path.stat().st_size for path in args.output.rglob("*") if path.is_file()
    )
    build_report["total_elapsed_seconds"] = time.perf_counter() - started
    (args.output / "build_report.json").write_text(
        json.dumps(build_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"paired mini dataset written to {args.output}: "
        f"{len(sample_index)} samples, {build_report['total_bytes']} bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
