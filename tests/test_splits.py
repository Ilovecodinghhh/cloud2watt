from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from cloud2watt.evaluation.splits import (
    assert_no_observation_overlap,
    build_split_manifest,
    temporal_split,
)


def synthetic_samples(days: int = 30, sites: int = 5) -> pd.DataFrame:
    issue_times = pd.date_range(
        datetime(2020, 1, 1, tzinfo=UTC) + timedelta(minutes=15),
        periods=days * 96,
        freq="15min",
    )
    records = []
    row = 0
    for site_id in range(sites):
        for issue_time in issue_times:
            records.append(
                {
                    "site_id": str(site_id),
                    "issue_time_utc": issue_time,
                    "power_history_row_indices": [row, row + 1, row + 2, row + 3],
                    "target_row_indices": [row + 4, row + 5, row + 6, row + 7, row + 8, row + 9],
                    "target_mask": [True] * 6,
                }
            )
            row += 10
    return pd.DataFrame(records)


def test_temporal_split_enforces_embargo_and_full_sample_containment() -> None:
    assigned, manifest = temporal_split(synthetic_samples(sites=1))
    assert {"train", "validation", "test", "embargo"}.issubset(set(assigned["temporal_split"]))
    train_last = assigned.loc[assigned["temporal_split"].eq("train"), "issue_time_utc"].max()
    validation_first = assigned.loc[
        assigned["temporal_split"].eq("validation"), "issue_time_utc"
    ].min()
    assert validation_first - timedelta(minutes=45) >= train_last + timedelta(minutes=240, hours=4)
    assert manifest["embargo_minutes"] == 240
    assert_no_observation_overlap(assigned)


def test_overlap_audit_detects_reused_physical_observation() -> None:
    samples = pd.DataFrame(
        {
            "temporal_split": ["train", "validation"],
            "power_history_row_indices": [[1], [1]],
            "target_row_indices": [[2], [3]],
        }
    )
    with pytest.raises(ValueError, match="both train and validation"):
        assert_no_observation_overlap(samples)


def test_split_manifest_is_deterministic_and_holds_out_twenty_percent() -> None:
    samples = synthetic_samples(sites=10)
    sites = pd.DataFrame(
        {
            "ss_id": range(10),
            "kWp": range(1, 11),
            "latitude_rounded": [50 + value / 10 for value in range(10)],
        }
    )
    first_assignments, first = build_split_manifest(
        samples, sites, data_manifest_hash="data", seed=7
    )
    second_assignments, second = build_split_manifest(
        samples, sites, data_manifest_hash="data", seed=7
    )
    assert first == second
    assert len(first["holdout_site_ids"]) == 2
    pd.testing.assert_frame_equal(first_assignments, second_assignments)
    assert first["test_locked_by_default"] is True
