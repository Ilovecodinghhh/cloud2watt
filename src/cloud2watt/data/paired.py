"""Build leakage-safe paired satellite and photovoltaic mini datasets."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import zarr

from cloud2watt.data.alignment import aggregate_power_15min, to_utc
from cloud2watt.data.quality import flag_power_quality
from cloud2watt.data.seviri import SeviriStore, SitePixel

HISTORY_MINUTES = (-45, -30, -15, 0)
FORECAST_MINUTES = (15, 30, 60, 120, 180, 240)


def datetime_index_to_ns(values: pd.DatetimeIndex) -> np.ndarray:
    """Return epoch nanoseconds independent of Pandas' internal resolution."""
    return values.to_numpy(dtype="datetime64[ns]").astype("int64")


def select_compact_sites(
    power: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    start_utc: datetime,
    end_utc: datetime,
    site_count: int = 20,
    spatial_bin_degrees: float = 0.2,
    minimum_coverage: float = 0.9,
) -> pd.DataFrame:
    """Choose well-covered sites in one compact deterministic spatial bin."""
    if site_count < 1 or spatial_bin_degrees <= 0:
        raise ValueError("site_count and spatial_bin_degrees must be positive")
    source = power.loc[:, ["ss_id", "datetime_GMT"]].copy()
    source["datetime_GMT"] = to_utc(source["datetime_GMT"])
    source = source.loc[
        source["datetime_GMT"].gt(start_utc) & source["datetime_GMT"].le(end_utc)
    ]
    expected = int((end_utc - start_utc).total_seconds() // 300)
    counts = source.groupby("ss_id").size().rename("reading_count").reset_index()
    candidates = counts.loc[counts["reading_count"] >= expected * minimum_coverage].merge(
        metadata,
        on="ss_id",
        how="inner",
        validate="one_to_one",
    )
    candidates["latitude_bin"] = (
        candidates["latitude_rounded"] / spatial_bin_degrees
    ).round() * spatial_bin_degrees
    candidates["longitude_bin"] = (
        candidates["longitude_rounded"] / spatial_bin_degrees
    ).round() * spatial_bin_degrees
    group_sizes = (
        candidates.groupby(["latitude_bin", "longitude_bin"])
        .size()
        .rename("site_count")
        .reset_index()
        .sort_values(
            ["site_count", "latitude_bin", "longitude_bin"],
            ascending=[False, True, True],
            kind="stable",
        )
    )
    if group_sizes.empty or int(group_sizes.iloc[0]["site_count"]) < site_count:
        raise ValueError("no compact spatial bin contains enough well-covered sites")
    selected_bin = group_sizes.iloc[0]
    selected = candidates.loc[
        candidates["latitude_bin"].eq(selected_bin["latitude_bin"])
        & candidates["longitude_bin"].eq(selected_bin["longitude_bin"])
    ].sort_values(["reading_count", "ss_id"], ascending=[False, True], kind="stable")
    columns = [
        "ss_id",
        "latitude_rounded",
        "longitude_rounded",
        "kWp",
        "orientation",
        "tilt",
        "reading_count",
    ]
    return selected.head(site_count).loc[:, columns].reset_index(drop=True)


def discover_paired_window(
    power: pd.DataFrame,
    metadata: pd.DataFrame,
    satellite_times: pd.DatetimeIndex,
    *,
    days: int = 7,
    site_count: int = 20,
) -> tuple[datetime, datetime, pd.DataFrame]:
    """Find the earliest full-day window with enough PV sites and satellite frames."""
    timestamps = to_utc(power["datetime_GMT"])
    first_day = timestamps.min().ceil("D")
    last_start = timestamps.max().floor("D") - pd.Timedelta(days=days)
    satellite_set = set(satellite_times)
    for candidate in pd.date_range(first_day, last_start, freq="D"):
        end = candidate + pd.Timedelta(days=days)
        required_power_start = candidate + pd.Timedelta(
            minutes=min(HISTORY_MINUTES) - 15
        )
        required_power_end = end + pd.Timedelta(minutes=max(FORECAST_MINUTES))
        if timestamps.min() > required_power_start or timestamps.max() < required_power_end:
            continue
        try:
            sites = select_compact_sites(
                power,
                metadata,
                start_utc=candidate.to_pydatetime(),
                end_utc=end.to_pydatetime(),
                site_count=site_count,
            )
        except ValueError:
            continue
        _, required_frames = paired_time_grid(
            candidate.to_pydatetime(), end.to_pydatetime()
        )
        if all(timestamp in satellite_set for timestamp in required_frames):
            return candidate.to_pydatetime(), end.to_pydatetime(), sites
    raise ValueError("no continuous paired window satisfies the requested duration and site count")


def prepare_paired_power(
    power: pd.DataFrame,
    sites: pd.DataFrame,
    *,
    issue_start_utc: datetime,
    issue_end_utc: datetime,
    value_semantics: Literal["interval_energy", "instantaneous_power"],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Prepare enough power history and targets around a requested issue window."""
    first_power_bin = issue_start_utc + timedelta(minutes=min(HISTORY_MINUTES) + 15)
    last_power_bin = issue_end_utc + timedelta(minutes=max(FORECAST_MINUTES))
    raw_start = first_power_bin - timedelta(minutes=15)
    source = power.copy()
    source["datetime_GMT"] = to_utc(source["datetime_GMT"])
    selected_ids = set(sites["ss_id"])
    source = source.loc[
        source["ss_id"].isin(selected_ids)
        & source["datetime_GMT"].gt(raw_start)
        & source["datetime_GMT"].le(last_power_bin)
    ]
    capacities = sites.set_index("ss_id")["kWp"]
    aligned = aggregate_power_15min(
        source,
        capacities,
        value_semantics=value_semantics,
    )
    flagged, stats = flag_power_quality(aligned)
    return flagged.reset_index(drop=True), stats.as_dict()


def paired_time_grid(
    issue_start_utc: datetime, issue_end_utc: datetime
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Return issue times and the deduplicated satellite input frame times."""
    issue_times = pd.date_range(
        issue_start_utc + timedelta(minutes=15), issue_end_utc, freq="15min"
    )
    satellite_times = pd.date_range(
        issue_times[0] + timedelta(minutes=min(HISTORY_MINUTES)),
        issue_times[-1],
        freq="15min",
    )
    return issue_times, satellite_times


def build_sample_index(
    power: pd.DataFrame,
    sites: pd.DataFrame,
    *,
    issue_times: pd.DatetimeIndex,
    satellite_times: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build references to deduplicated frames and power rows for every sample."""
    power = power.reset_index(drop=True).copy()
    power_lookup = {
        (str(row.ss_id), pd.Timestamp(row.datetime_GMT)): index
        for index, row in enumerate(power.itertuples(index=False))
    }
    satellite_lookup = {timestamp: index for index, timestamp in enumerate(satellite_times)}
    records = []
    dropped_history = 0
    for site_index, site in enumerate(sites.itertuples(index=False)):
        site_id = str(site.ss_id)
        for issue_time in issue_times:
            history_times = [
                issue_time + timedelta(minutes=offset) for offset in HISTORY_MINUTES
            ]
            target_times = [
                issue_time + timedelta(minutes=offset) for offset in FORECAST_MINUTES
            ]
            power_history = [power_lookup.get((site_id, time), -1) for time in history_times]
            if any(index < 0 or not bool(power.iloc[index]["is_valid"]) for index in power_history):
                dropped_history += 1
                continue
            satellite_indices = [satellite_lookup.get(time, -1) for time in history_times]
            if any(index < 0 for index in satellite_indices):
                raise ValueError("satellite history grid is incomplete")
            target_indices = [power_lookup.get((site_id, time), -1) for time in target_times]
            target_mask = [
                index >= 0 and bool(power.iloc[index]["is_valid"]) for index in target_indices
            ]
            records.append(
                {
                    "site_id": site_id,
                    "site_index": site_index,
                    "issue_time_utc": issue_time,
                    "satellite_frame_indices": satellite_indices,
                    "power_history_row_indices": power_history,
                    "target_row_indices": target_indices,
                    "target_mask": target_mask,
                    "quality_flags": "" if all(target_mask) else "incomplete_target",
                }
            )
    return pd.DataFrame.from_records(records), {"dropped_invalid_history": dropped_history}


def match_satellite_indices(
    requested: pd.DatetimeIndex, available: pd.DatetimeIndex
) -> list[int]:
    """Match exact UTC frame times and fail instead of selecting future data."""
    if available.has_duplicates:
        raise ValueError("satellite time coordinate contains duplicates")
    lookup = {timestamp: index for index, timestamp in enumerate(available)}
    missing = [timestamp for timestamp in requested if timestamp not in lookup]
    if missing:
        preview = ", ".join(str(value) for value in missing[:3])
        raise ValueError(f"missing {len(missing)} required satellite frames: {preview}")
    return [lookup[timestamp] for timestamp in requested]


def validate_sample_index(
    sample_index: pd.DataFrame,
    power: pd.DataFrame,
    satellite_times: pd.DatetimeIndex,
) -> dict[str, int]:
    """Validate every temporal reference and return an audit summary."""
    if sample_index.duplicated(["site_id", "issue_time_utc"]).any():
        raise ValueError("duplicate (site_id, issue_time_utc) sample keys")
    checked_satellite_frames = 0
    checked_power_references = 0
    for row in sample_index.itertuples(index=False):
        issue_time = pd.Timestamp(row.issue_time_utc)
        expected_history = [
            issue_time + timedelta(minutes=offset) for offset in HISTORY_MINUTES
        ]
        actual_satellite = [satellite_times[int(index)] for index in row.satellite_frame_indices]
        if actual_satellite != expected_history or max(actual_satellite) > issue_time:
            raise ValueError("satellite history is misaligned or contains future data")
        actual_power_history = [
            pd.Timestamp(power.iloc[int(index)]["datetime_GMT"])
            for index in row.power_history_row_indices
        ]
        if actual_power_history != expected_history:
            raise ValueError("power history is misaligned")
        expected_targets = [
            issue_time + timedelta(minutes=offset) for offset in FORECAST_MINUTES
        ]
        for target_index, expected_time in zip(
            row.target_row_indices, expected_targets, strict=True
        ):
            if int(target_index) >= 0:
                actual_time = pd.Timestamp(power.iloc[int(target_index)]["datetime_GMT"])
                if actual_time != expected_time or actual_time <= issue_time:
                    raise ValueError("target is misaligned or overlaps model inputs")
                checked_power_references += 1
        checked_satellite_frames += len(actual_satellite)
        checked_power_references += len(actual_power_history)
    return {
        "validated_samples": len(sample_index),
        "validated_satellite_references": checked_satellite_frames,
        "validated_power_references": checked_power_references,
    }


def write_satellite_frames(
    output_path: Path,
    client: SeviriStore,
    site_pixels: list[SitePixel],
    *,
    requested_times: pd.DatetimeIndex,
    source_time_indices: list[int],
    channel_names: list[str],
    height: int = 128,
    width: int = 128,
) -> dict[str, object]:
    """Write unique site/time frames to chunked Zarr without history duplication."""
    available_channels = client.channels()
    channel_indices = [available_channels.index(name) for name in channel_names]
    required_keys = client.required_data_keys(
        source_time_indices, site_pixels, height=height, width=width
    )
    started = time.perf_counter()
    client.prefetch(required_keys)
    root = zarr.open_group(output_path, mode="w")
    frames = root.create_array(
        "frames",
        shape=(len(site_pixels), len(requested_times), len(channel_names), height, width),
        chunks=(1, 12, len(channel_names), height, width),
        dtype="float16",
    )
    root.create_array(
        "time_ns",
        data=datetime_index_to_ns(requested_times),
        chunks=(min(len(requested_times), 1024),),
    )
    root.attrs.update(
        {
            "dimensions": ["site", "time", "channel", "y", "x"],
            "site_ids": [site.site_id for site in site_pixels],
            "channel_names": channel_names,
            "source_url": client.store_url,
            "source_metadata_sha256": client.metadata_sha256,
        }
    )
    finite_count = 0
    value_count = 0
    for time_start in range(0, len(requested_times), 12):
        time_stop = min(time_start + 12, len(requested_times))
        block = np.empty(
            (
                len(site_pixels),
                time_stop - time_start,
                len(channel_names),
                height,
                width,
            ),
            dtype=np.float16,
        )
        for local_time, source_index in enumerate(
            source_time_indices[time_start:time_stop]
        ):
            for site_index, site in enumerate(site_pixels):
                block[site_index, local_time] = client.read_crop(
                    time_index=source_index,
                    site=site,
                    channel_indices=channel_indices,
                    height=height,
                    width=width,
                )
        finite_count += int(np.isfinite(block).sum())
        value_count += block.size
        frames[:, time_start:time_stop] = block
    elapsed = time.perf_counter() - started
    return {
        "required_remote_chunks": len(required_keys),
        "network_fetches": client.network_fetches,
        "network_retries": client.retry_count,
        "cache_hits": client.cache_hits,
        "downloaded_bytes": client.downloaded_bytes,
        "elapsed_seconds": elapsed,
        "finite_fraction": finite_count / value_count,
    }


def write_paired_metadata(
    output_dir: Path,
    *,
    power: pd.DataFrame,
    sites: pd.DataFrame,
    sample_index: pd.DataFrame,
    manifest: dict[str, object],
    build_report: dict[str, object],
) -> None:
    power.to_parquet(output_dir / "power_15min.parquet", index=False)
    sites.to_parquet(output_dir / "sites.parquet", index=False)
    sample_index.to_parquet(output_dir / "sample_index.parquet", index=False)
    for name, payload in (("manifest.json", manifest), ("build_report.json", build_report)):
        (output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def hash_sample_keys(sample_index: pd.DataFrame) -> str:
    """Hash the ordered sample-key set for reproducibility checks."""
    values = "\n".join(
        f"{row.site_id},{pd.Timestamp(row.issue_time_utc).isoformat()}"
        for row in sample_index.itertuples(index=False)
    )
    return hashlib.sha256(values.encode("utf-8")).hexdigest()


class PairedDataset:
    """Small local reader that assembles samples from deduplicated frame storage."""

    def __init__(self, root: Path) -> None:
        self.sample_index = pd.read_parquet(root / "sample_index.parquet")
        self.power = pd.read_parquet(root / "power_15min.parquet")
        self.frames = zarr.open_group(root / "satellite_frames.zarr", mode="r")["frames"]

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.sample_index.iloc[index]
        satellite = np.stack(
            [self.frames[int(row.site_index), int(frame)] for frame in row.satellite_frame_indices]
        )
        history_rows = np.asarray(row.power_history_row_indices, dtype=int)
        target_rows = np.asarray(row.target_row_indices, dtype=int)
        target = np.full(len(target_rows), np.nan, dtype=np.float32)
        valid_target = target_rows >= 0
        target[valid_target] = self.power.iloc[target_rows[valid_target]][
            "normalized_power"
        ].to_numpy(dtype=np.float32)
        return {
            "site_id": row.site_id,
            "issue_time_utc": row.issue_time_utc,
            "satellite": satellite,
            "power_history": self.power.iloc[history_rows]["normalized_power"].to_numpy(
                dtype=np.float32
            ),
            "target_power": target,
            "target_mask": np.asarray(row.target_mask, dtype=bool),
        }
