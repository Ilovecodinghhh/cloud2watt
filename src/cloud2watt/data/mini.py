"""Build a bounded seven-day UK PV dataset for pipeline development."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd

from cloud2watt.data.alignment import aggregate_power_15min, to_utc
from cloud2watt.data.manifest import build_fixture_manifest
from cloud2watt.data.quality import QualityFlag, flag_power_quality


def build_power_mini_dataset(
    power: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    start_utc: datetime,
    days: int = 7,
    site_count: int = 20,
    bad_periods: pd.DataFrame | None = None,
    source_revision: str = "unknown",
    power_value_semantics: Literal["interval_energy", "instantaneous_power"],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Select well-covered sites, align power to 15 minutes, and return a manifest."""
    if start_utc.tzinfo is None or start_utc.utcoffset() != timedelta(0):
        raise ValueError("start_utc must be timezone-aware UTC")
    if days < 1 or site_count < 1:
        raise ValueError("days and site_count must be positive")
    required_metadata = {"ss_id", "latitude_rounded", "longitude_rounded", "kWp"}
    missing = required_metadata.difference(metadata.columns)
    if missing:
        raise ValueError(f"missing metadata columns: {sorted(missing)}")

    end_utc = start_utc + timedelta(days=days)
    source = power.copy()
    source["datetime_GMT"] = to_utc(source["datetime_GMT"])
    source = source.loc[source["datetime_GMT"].gt(start_utc) & source["datetime_GMT"].le(end_utc)]
    counts = source.groupby("ss_id", sort=True).size().rename("reading_count")
    ranking = counts.reset_index().sort_values(
        ["reading_count", "ss_id"], ascending=[False, True], kind="stable"
    )
    selected_ids = ranking.head(site_count)["ss_id"].tolist()
    if len(selected_ids) < site_count:
        raise ValueError(f"only {len(selected_ids)} sites overlap the requested interval")

    sites = (
        metadata.loc[metadata["ss_id"].isin(selected_ids), sorted(required_metadata)]
        .drop_duplicates("ss_id")
        .set_index("ss_id")
        .reindex(selected_ids)
        .reset_index()
    )
    if sites["kWp"].isna().any():
        raise ValueError("selected sites are missing metadata")
    capacities = sites.set_index("ss_id")["kWp"]
    aligned = aggregate_power_15min(
        source.loc[source["ss_id"].isin(selected_ids)],
        capacities,
        value_semantics=power_value_semantics,
    )
    flagged, stats = flag_power_quality(aligned)
    if bad_periods is not None:
        flagged = mark_source_bad_periods(flagged, bad_periods)
        source_bad_count = int(
            (flagged["quality_flags"].astype("uint16") & int(QualityFlag.SOURCE_BAD_PERIOD))
            .ne(0)
            .sum()
        )
        stats.record(QualityFlag.SOURCE_BAD_PERIOD, source_bad_count)
    manifest = build_fixture_manifest(
        [str(site_id) for site_id in selected_ids], start_utc=start_utc, days=days
    )
    manifest.update(
        {
            "source": "openclimatefix/uk_pv",
            "source_revision": source_revision,
            "selection": "highest 5-minute coverage, ties ordered by ss_id",
            "quality_counts": stats.as_dict(),
            "rows": len(flagged),
            "valid_rows": int(flagged["is_valid"].sum()),
            "power_value_semantics": power_value_semantics,
        }
    )
    return flagged, sites, manifest


def write_power_mini_dataset(
    output_dir: Path,
    power: pd.DataFrame,
    sites: pd.DataFrame,
    manifest: dict[str, object],
) -> None:
    """Persist derived tables and their selection/provenance manifest."""
    output_dir.mkdir(parents=True, exist_ok=False)
    power.to_parquet(output_dir / "power_15min.parquet", index=False)
    sites.to_parquet(output_dir / "sites.parquet", index=False)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def mark_source_bad_periods(power: pd.DataFrame, bad_periods: pd.DataFrame) -> pd.DataFrame:
    """Add the source bad-period bit for inclusive site-specific UTC intervals."""
    result = power.copy()
    result["datetime_GMT"] = to_utc(result["datetime_GMT"])
    if "quality_flags" not in result:
        result["quality_flags"] = 0
    for row in bad_periods.itertuples(index=False):
        start = pd.to_datetime(row.start_datetime_GMT, utc=True)
        end_value = getattr(row, "end_datetime_GMT", None)
        end = (
            pd.Timestamp.max.tz_localize("UTC")
            if pd.isna(end_value)
            else pd.to_datetime(end_value, utc=True)
        )
        mask = (
            result["ss_id"].eq(row.ss_id)
            & result["datetime_GMT"].ge(start)
            & result["datetime_GMT"].le(end)
        )
        result.loc[mask, "quality_flags"] = result.loc[mask, "quality_flags"].astype(
            "uint16"
        ) | int(QualityFlag.SOURCE_BAD_PERIOD)
    result["is_valid"] = result["quality_flags"].eq(int(QualityFlag.OK))
    return result
