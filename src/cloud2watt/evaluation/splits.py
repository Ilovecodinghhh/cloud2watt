"""Leakage-safe temporal and spatial dataset splits."""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta

import numpy as np
import pandas as pd

from cloud2watt.data.paired import FORECAST_MINUTES, HISTORY_MINUTES


def _utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("split timestamps must be timezone-aware")
    return timestamp.tz_convert("UTC")


def temporal_split(
    samples: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    embargo: timedelta = timedelta(hours=4),
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Assign samples only when their complete input/target interval fits one split."""
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("fractions must leave non-empty train, validation, and test ranges")
    result = samples.copy()
    times = pd.to_datetime(result["issue_time_utc"], utc=True)
    start, end = times.min(), times.max()
    duration = end - start
    train_end = (start + duration * train_fraction).floor("15min")
    validation_end = (start + duration * (train_fraction + validation_fraction)).floor(
        "15min"
    )
    sample_start = times + pd.Timedelta(minutes=min(HISTORY_MINUTES))
    sample_end = times + pd.Timedelta(minutes=max(FORECAST_MINUTES))
    labels = pd.Series("embargo", index=result.index, dtype="string")
    labels.loc[sample_end.le(train_end)] = "train"
    validation_start = train_end + embargo
    labels.loc[sample_start.ge(validation_start) & sample_end.le(validation_end)] = "validation"
    test_start = validation_end + embargo
    labels.loc[sample_start.ge(test_start)] = "test"
    result["temporal_split"] = labels
    boundaries = {
        "data_start_utc": start.isoformat(),
        "train_end_utc": train_end.isoformat(),
        "validation_start_utc": validation_start.isoformat(),
        "validation_end_utc": validation_end.isoformat(),
        "test_start_utc": test_start.isoformat(),
        "data_end_utc": end.isoformat(),
        "embargo_minutes": int(embargo.total_seconds() // 60),
        "counts": {str(key): int(value) for key, value in labels.value_counts().items()},
    }
    return result, boundaries


def assert_no_observation_overlap(samples: pd.DataFrame) -> None:
    """Fail if any referenced power row appears in more than one temporal split."""
    seen: dict[int, str] = {}
    for row in samples.loc[samples["temporal_split"].ne("embargo")].itertuples(index=False):
        references = list(row.power_history_row_indices) + [
            int(value) for value in row.target_row_indices if int(value) >= 0
        ]
        for reference in references:
            previous = seen.setdefault(int(reference), str(row.temporal_split))
            if previous != row.temporal_split:
                raise ValueError(
                    f"power observation {reference} occurs in both {previous} "
                    f"and {row.temporal_split}"
                )


def stratified_site_holdout(
    sites: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    fraction: float = 0.20,
    seed: int = 42,
) -> list[str]:
    """Select a deterministic holdout balanced over capacity, latitude, and validity."""
    if not 0 < fraction < 1:
        raise ValueError("holdout fraction must be between zero and one")
    valid_rate = (
        samples.assign(_valid=samples["target_mask"].map(lambda values: float(np.mean(values))))
        .groupby("site_id")["_valid"]
        .mean()
    )
    frame = sites.copy()
    frame["site_id"] = frame["ss_id"].astype(str)
    frame["valid_rate"] = frame["site_id"].map(valid_rate).fillna(0.0)
    for source, target in (
        ("kWp", "capacity_bin"),
        ("latitude_rounded", "latitude_bin"),
        ("valid_rate", "validity_bin"),
    ):
        frame[target] = pd.qcut(
            frame[source], q=min(3, len(frame)), labels=False, duplicates="drop"
        )
    frame["stratum"] = (
        frame[["capacity_bin", "latitude_bin", "validity_bin"]]
        .fillna(-1)
        .astype(int)
        .astype(str)
        .agg("-".join, axis=1)
    )
    frame["rank"] = frame["site_id"].map(
        lambda site_id: hashlib.sha256(f"{seed}:{site_id}".encode()).hexdigest()
    )
    ordered = frame.sort_values(["stratum", "rank", "site_id"], kind="stable")
    ordered["within_stratum_rank"] = ordered.groupby("stratum").cumcount()
    ordered = ordered.sort_values(
        ["within_stratum_rank", "stratum", "rank"], kind="stable"
    )
    count = max(1, round(len(frame) * fraction))
    return ordered.head(count)["site_id"].sort_values().tolist()


def build_split_manifest(
    samples: pd.DataFrame,
    sites: pd.DataFrame,
    *,
    data_manifest_hash: str,
    seed: int = 42,
    embargo_hours: int = 4,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build assignments and an immutable, self-hashed split manifest."""
    assigned, boundaries = temporal_split(samples, embargo=timedelta(hours=embargo_hours))
    assert_no_observation_overlap(assigned)
    holdout = stratified_site_holdout(sites, assigned, seed=seed)
    assigned["site_split"] = np.where(
        assigned["site_id"].astype(str).isin(holdout), "holdout", "development"
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "data_manifest_hash": data_manifest_hash,
        "fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "boundaries": boundaries,
        "history_minutes": list(HISTORY_MINUTES),
        "forecast_minutes": list(FORECAST_MINUTES),
        "holdout_fraction": 0.20,
        "holdout_site_ids": holdout,
        "random_seed": seed,
        "test_locked_by_default": True,
    }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["split_manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    return assigned, manifest
