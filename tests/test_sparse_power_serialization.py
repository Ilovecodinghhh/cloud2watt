"""Missing source bins retain QC and a Parquet-safe duplicate flag."""

import pandas as pd

from cloud2watt.data.paired import prepare_paired_power


def test_missing_bins_can_be_persisted_without_erasing_missing_quality(tmp_path):
    times = pd.to_datetime(
        [
            "2021-01-01T00:05Z",
            "2021-01-01T00:10Z",
            "2021-01-01T00:15Z",
            "2021-01-01T01:05Z",
            "2021-01-01T01:10Z",
            "2021-01-01T01:15Z",
        ]
    )
    raw = pd.DataFrame({"ss_id": ["a"] * 6, "datetime_GMT": times, "generation_Wh": [10.0] * 6})
    sites = pd.DataFrame({"ss_id": ["a"], "kWp": [1.0]})
    power, _ = prepare_paired_power(
        raw,
        sites,
        issue_start_utc=pd.Timestamp("2021-01-01T00:45Z"),
        issue_end_utc=pd.Timestamp("2021-01-01T01:00Z"),
        value_semantics="interval_energy",
    )
    destination = tmp_path / "power.parquet"
    power.to_parquet(destination, index=False)
    restored = pd.read_parquet(destination)
    assert restored.raw_duplicate.dtype == bool
    missing = restored.reading_count.eq(0)
    assert missing.any()
    assert not restored.loc[missing, "is_valid"].any()
    assert restored.loc[missing, "normalized_power"].isna().all()
