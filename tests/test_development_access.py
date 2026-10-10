"""Development label reads must exclude future/test time windows."""
import numpy as np
import pandas as pd
import pytest

from cloud2watt.data.development import read_power_for_samples


def test_filter_before_label_materialization_preserves_positional_references(tmp_path, monkeypatch):
    times = pd.date_range("2021-01-01", periods=8, freq="15min", tz="UTC")
    source = pd.DataFrame({"ss_id": ["a"] * 8, "datetime_GMT": times,
                           "normalized_power": np.arange(8) / 10})
    path = tmp_path / "power.parquet"
    source.to_parquet(path, index=False)
    samples = pd.DataFrame({"power_history_row_indices": [[1, 2]],
                            "target_row_indices": [[3, 4, -1]]})
    original = pd.read_parquet
    calls = []

    def recording_read(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", recording_read)
    result = read_power_for_samples(path, samples)
    assert calls[0]["columns"] == ["ss_id", "datetime_GMT"]
    assert calls[1]["filters"][-1] == ("datetime_GMT", "<=", times[4])
    assert result.normalized_power.iloc[5:].isna().all()
    np.testing.assert_allclose(result.normalized_power.iloc[1:5], [0.1, 0.2, 0.3, 0.4])
    assert result.datetime_GMT.equals(source.datetime_GMT)
    samples.at[0, "target_row_indices"] = [99]
    with pytest.raises(ValueError, match="invalid power"):
        read_power_for_samples(path, samples)
