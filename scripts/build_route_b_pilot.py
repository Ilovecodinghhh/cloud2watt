"""Build a bounded route B development pilot without opening final-test labels."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from huggingface_hub import hf_hub_download

from cloud2watt.data.bounded_source import BoundedSeviriStore
from cloud2watt.data.mini import mark_source_bad_periods
from cloud2watt.data.paired import (
    FORECAST_MINUTES,
    HISTORY_MINUTES,
    build_sample_index,
    datetime_index_to_ns,
    hash_sample_keys,
    partition_satellite_times,
    prepare_paired_power,
    validate_sample_index,
)
from cloud2watt.data.quality import QualityFlag, flag_context_quality, quality_breakdown
from cloud2watt.data.seviri import SEVIRI_PROJ4, SitePixel
from cloud2watt.data.shared_tiles import SCHEMA, SharedFrames, intersections, store_bytes
from cloud2watt.data.solar import build_solar_features
from cloud2watt.run_state import atomic_json, file_hash, json_hash


def pilot_bounds(preflight, region_id, window_id, start, end):
    region = next(r for r in preflight["regions"] if r["region_id"] == region_id)
    window = next(w for w in preflight["windows"] if w["id"] == window_id)
    if region["role"] != "development" or window["role"] != "development":
        raise ValueError("pilot refuses locked geography and final-test windows")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("explicit UTC boundaries required")
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    if not pd.Timestamp(window["start"]) <= start < end <= pd.Timestamp(window["end"]):
        raise ValueError("pilot outside frozen development window")
    if not pd.Timedelta(hours=6) <= end - start <= pd.Timedelta(days=2):
        raise ValueError("engineering pilot must span 6 hours to 2 days")
    # All raw history and target support stays inside [start, end).
    issues = pd.date_range(
        start + pd.Timedelta(hours=1), end - pd.Timedelta(hours=4, minutes=15), freq="15min"
    )
    return region, start, end, issues


def build(args):
    started = time.perf_counter()
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    region, start, end, issues = pilot_bounds(
        preflight, args.region, args.window, args.start, args.end
    )
    power_issues = issues.copy()
    if getattr(args, "issue_start", None) is not None:
        issues = issues[
            (issues >= pd.Timestamp(args.issue_start)) & (issues < pd.Timestamp(args.issue_end))
        ]
        if not len(issues):
            raise ValueError("empty partition issue range")
    role = getattr(args, "data_role", "engineering_development_only")
    guard = getattr(args, "guard", lambda: None)
    guard()
    if args.output.exists():
        raise FileExistsError("pilot outputs are immutable; choose a fresh output")
    disk = shutil.disk_usage(args.output.parent)
    if disk.free - 2 * 1024**3 < 0.2 * disk.total:
        raise ValueError("pilot needs 2 GiB above disk reserve")
    candidates_path = args.preflight.parent / "candidate_sites.csv"
    candidates = pd.read_csv(candidates_path, dtype={"ss_id": str})
    sites = candidates.set_index("ss_id").loc[region["site_ids"]].reset_index()
    source = preflight["satellite_sources"][str(start.year)]
    args.output.mkdir(parents=True, exist_ok=False)
    prereg = {
        "role": role,
        "issue_start_inclusive": issues[0].isoformat(),
        "issue_end_exclusive": (issues[-1] + pd.Timedelta(minutes=15)).isoformat(),
        "region": args.region,
        "window": args.window,
        "support_start_inclusive": start.isoformat(),
        "support_end_exclusive": end.isoformat(),
        "site_ids": region["site_ids"],
        "preflight_sha256": file_hash(args.preflight),
        "candidate_sites_sha256": file_hash(candidates_path),
        "builder_sha256": file_hash(Path(__file__)),
        "source_hashes": {
            str(p): file_hash(p) for p in sorted(Path("src/cloud2watt").rglob("*.py"))
        },
        "source_cache_bytes": 4 * 1024**3,
        "cumulative_transfer_limit": 100 * 1024**3,
        "label_contract": "c2w-sampled-power-v1",
        "test_labels_read": False,
        "channels": ["VIS006", "IR_016", "IR_108"],
        "crop_shape": [128, 128],
        "satellite_invalid_fraction_limit": 0.05,
        "evaluation_protocol": "engineering_only_not_c2w-eval-v2.2",
    }
    atomic_json(args.output / "preregistration.json", prereg)
    client = getattr(args, "client", None)
    if client is None:
        client = BoundedSeviriStore(source["url"], cache_dir=args.cache, max_attempts=3)
    downloaded_before = client.downloaded_bytes
    if client.metadata_sha256 != source["metadata_sha256"]:
        raise ValueError("source metadata drift")
    months = pd.period_range(
        start.tz_localize(None), (end - pd.Timedelta(seconds=1)).tz_localize(None), freq="M"
    )
    raw_frames, inputs = [], []
    for month in months:
        name = f"5_minutely/year={month.year}/month={month.month:02d}/data.parquet"
        path = Path(
            hf_hub_download(
                "openclimatefix/uk_pv",
                name,
                repo_type="dataset",
                revision=preflight["pv_revision"],
                cache_dir=args.preflight.parent / "pv_source",
                local_files_only=getattr(args, "local_sources_only", False),
            )
        )
        raw_frames.append(
            pd.read_parquet(
                path,
                filters=[
                    ("ss_id", "in", [int(i) for i in region["site_ids"]]),
                    ("datetime_GMT", ">=", start),
                    ("datetime_GMT", "<", end),
                ],
            )
        )
        inputs.append({"partition": name, "sha256": file_hash(path)})
    raw = pd.concat(raw_frames, ignore_index=True)
    raw["ss_id"] = raw.ss_id.astype(str)
    if raw.empty:
        raise ValueError("no pilot PV records")
    power, _ = prepare_paired_power(
        raw,
        sites,
        issue_start_utc=power_issues[0].to_pydatetime() - pd.Timedelta(minutes=15),
        issue_end_utc=power_issues[-1].to_pydatetime(),
        value_semantics="interval_energy",
    )
    # Legacy arithmetic is three scaled samples, not verified interval energy.
    solar = build_solar_features(pd.DatetimeIndex(power.datetime_GMT.unique()).sort_values(), sites)
    power = power.merge(solar, on=["ss_id", "datetime_GMT"], validate="one_to_one")
    power, _ = flag_context_quality(
        power,
        night_elevation_degrees=0,
        night_nonzero_fraction=0.01,
        daytime_elevation_degrees=10,
        stuck_zero_bins=8,
    )
    probe = json.loads(Path("outputs/probes/uk_pv.json").read_text(encoding="utf-8"))
    bad_path = Path(probe["details"]["bad_data"]["path"])
    if file_hash(bad_path) != probe["details"]["bad_data"]["sha256"]:
        raise ValueError("bad-period source identity changed")
    power = mark_source_bad_periods(power, pd.read_csv(bad_path, dtype={"ss_id": str}))
    power = power.reset_index(drop=True)
    requested = pd.date_range(issues[0] - pd.Timedelta(minutes=45), issues[-1], freq="15min")
    times, source_indices, missing = partition_satellite_times(requested, client.timestamps())
    if not len(times):
        raise ValueError("no source satellite frames")
    pixels = client.map_sites(sites)
    sites["satellite_x_index"] = [p.x_index for p in pixels]
    sites["satellite_y_index"] = [p.y_index for p in pixels]
    origins = [(p.y_index - 64, p.x_index - 64) for p in pixels]
    y, x = np.min(origins, axis=0).tolist()
    y1, x1 = (np.max(origins, axis=0) + [128, 128]).tolist()
    if (y1 - y) * (x1 - x) > 256**2:
        raise ValueError("pilot region bounding box exceeds budget")
    bbox = SitePixel("region", x + (x1 - x) // 2, y + (y1 - y) // 2, 0, 0)
    channels = [client.channels().index(c) for c in prereg["channels"]]
    keys = client.required_data_keys(source_indices, [bbox], height=y1 - y, width=x1 - x)
    atomic_json(
        args.output / "source_plan.json",
        {
            "required_data_objects": len(keys),
            "object_keys": keys,
            "pv_inputs": inputs,
            "missing_times": missing.astype(str).tolist(),
        },
    )
    root = zarr.open_group(args.output / "satellite_frames.zarr", mode="w")
    tiles = sorted(
        {(p[0], p[1]) for oy, ox in origins for p in intersections(oy, ox, 128, 128, 128)}
    )
    root.attrs.update(
        {
            "schema": SCHEMA,
            "complete": False,
            "tile_size": 128,
            "time_chunk": 12,
            "site_origins": origins,
            "tiles": tiles,
            "site_ids": region["site_ids"],
            "crop_shape": [128, 128],
            "channel_names": prereg["channels"],
            "projection_proj4": SEVIRI_PROJ4,
            "geometry": "corrected_sweep_y",
        }
    )
    root.create_array("time_ns", data=datetime_index_to_ns(times))
    root.create_array("source_frame_indices", data=np.asarray(source_indices))
    arrays, coverage = {}, {}
    for ty, tx in tiles:
        arrays[ty, tx] = root.create_array(
            f"tiles/{ty}_{tx}",
            shape=(len(times), 3, 128, 128),
            chunks=(12, 3, 128, 128),
            dtype="float16",
            fill_value=np.nan,
        )
        mask = np.zeros((128, 128), dtype=bool)
        for oy, ox in origins:
            for py, px, a, b, c, d in intersections(oy, ox, 128, 128, 128):
                if (py, px) == (ty, tx):
                    mask[a - ty : b - ty, c - tx : d - tx] = True
        coverage[ty, tx] = mask
        root.create_array(f"coverage/{ty}_{tx}", data=mask)
    records = []
    for first in range(0, len(times), 12):
        guard()
        last = min(first + 12, len(times))
        block = np.stack(
            [
                client.read_crop(
                    time_index=int(i),
                    site=bbox,
                    channel_indices=channels,
                    height=y1 - y,
                    width=x1 - x,
                )
                for i in source_indices[first:last]
            ]
        )
        for si, (oy, ox) in enumerate(origins):
            invalid = 1 - np.isfinite(
                block[:, :, oy - y : oy - y + 128, ox - x : ox - x + 128]
            ).mean((1, 2, 3))
            for slot, fraction in enumerate(invalid):
                records.append(
                    {
                        "ss_id": region["site_ids"][si],
                        "datetime_GMT": times[first + slot],
                        "invalid_fraction": float(fraction),
                        "quality_flags": int(QualityFlag.SATELLITE_INVALID)
                        if fraction > 0.05
                        else 0,
                        "is_valid": bool(fraction <= 0.05),
                    }
                )
        for ty, tx, a, b, c, d in intersections(y, x, y1 - y, x1 - x, 128):
            if (ty, tx) not in arrays:
                continue
            part = np.full((last - first, 3, 128, 128), np.nan, dtype="float16")
            part[:, :, a - ty : b - ty, c - tx : d - tx] = block[:, :, a - y : b - y, c - x : d - x]
            part[:, :, ~coverage[ty, tx]] = np.nan
            arrays[ty, tx][first:last] = part
        print(
            f"satellite frames {last}/{len(times)}; downloaded {client.downloaded_bytes}",
            flush=True,
        )
    for site_id in region["site_ids"]:
        for timestamp in missing:
            records.append(
                {
                    "ss_id": site_id,
                    "datetime_GMT": timestamp,
                    "invalid_fraction": 1.0,
                    "quality_flags": int(QualityFlag.SATELLITE_MISSING),
                    "is_valid": False,
                }
            )
    quality = pd.DataFrame(records)
    invalid_keys = {(r.ss_id, r.datetime_GMT) for r in quality.loc[~quality.is_valid].itertuples()}
    samples, stats = build_sample_index(
        power, sites, issue_times=issues, satellite_times=times, invalid_satellite_keys=invalid_keys
    )
    if samples.empty:
        raise ValueError("pilot has no paired samples")
    validation = validate_sample_index(samples, power, times)
    if not (
        (samples.issue_time_utc - pd.Timedelta(hours=1) >= start).all()
        and (samples.issue_time_utc + pd.Timedelta(hours=4) < end).all()
    ):
        raise ValueError("sample raw support escaped development boundaries")
    root.attrs["complete"] = True
    reader = SharedFrames(root)
    for si, pixel in enumerate(pixels):
        for slot in (0, len(times) // 2, len(times) - 1):
            expected = client.read_crop(
                time_index=source_indices[slot],
                site=pixel,
                channel_indices=channels,
                height=128,
                width=128,
            )
            np.testing.assert_array_equal(reader[si, slot], expected)
    power.to_parquet(args.output / "power_15min.parquet", index=False)
    sites.to_parquet(args.output / "sites.parquet", index=False)
    samples.to_parquet(args.output / "sample_index.parquet", index=False)
    quality.to_parquet(args.output / "satellite_quality.parquet", index=False)
    manifest = {
        "schema_version": 3,
        "dataset_version": args.output.name,
        "data_role": role,
        "source": {
            "pv": "openclimatefix/uk_pv",
            "pv_revision": preflight["pv_revision"],
            "satellite": source["url"],
            "satellite_metadata_sha256": client.metadata_sha256,
        },
        "site_count": len(sites),
        "site_ids": region["site_ids"],
        "sample_count": len(samples),
        "sample_key_sha256": hash_sample_keys(samples),
        "channels": prereg["channels"],
        "crop_shape": [128, 128],
        "history_minutes": list(HISTORY_MINUTES),
        "forecast_minutes": list(FORECAST_MINUTES),
        "label_contract": "c2w-sampled-power-v1",
        "power_value_semantics": "mean_of_three_scaled_instantaneous_samples",
        "satellite_layout": SCHEMA,
        "projection_proj4": SEVIRI_PROJ4,
        "availability_basis": "idealized_archive_time",
        "evaluation_protocol": prereg["evaluation_protocol"],
        "preregistration_sha256": file_hash(args.output / "preregistration.json"),
    }
    manifest["manifest_content_sha256"] = json_hash(manifest)
    atomic_json(args.output / "manifest.json", manifest)
    report = {
        "status": "pilot_complete_not_formal_training_ready",
        "samples": len(samples),
        "pv_rows": len(power),
        "raw_pv_rows": len(raw),
        "sites": len(sites),
        "quality": quality_breakdown(power),
        "satellite_quality": quality_breakdown(quality),
        "sample_stats": stats,
        "validation": validation,
        "direct_source_crop_checks": len(sites) * 3,
        "stored_frames": len(times),
        "missing_frames": len(missing),
        "required_source_objects": len(keys),
        "downloaded_bytes": client.downloaded_bytes - downloaded_before,
        "evicted_bytes": client.evicted_bytes,
        "cache_bytes": store_bytes(client.objects),
        "derived_bytes": store_bytes(args.output),
        "elapsed_seconds": time.perf_counter() - started,
        "test_labels_read": False,
    }
    atomic_json(args.output / "build_report.json", report)
    atomic_json(
        args.output / "complete.json",
        {
            "manifest_sha256": file_hash(args.output / "manifest.json"),
            "build_report_sha256": file_hash(args.output / "build_report.json"),
            "table_hashes": {
                n: file_hash(args.output / n)
                for n in (
                    "power_15min.parquet",
                    "sample_index.parquet",
                    "sites.parquet",
                    "satellite_quality.parquet",
                )
            },
        },
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--region", default="r0")
    parser.add_argument("--window", default="winter_fit")
    parser.add_argument("--start", default="2021-01-01T00:00Z")
    parser.add_argument("--end", default="2021-01-03T00:00Z")
    parser.add_argument("--cache", type=Path, default=Path("outputs/project-v2/source-cache-2021"))
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args())
