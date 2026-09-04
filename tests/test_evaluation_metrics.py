from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cloud2watt.baselines import SiteTimeClimatology, smart_persistence
from cloud2watt.evaluation.metrics import (
    event_precision_recall_f1,
    masked_mae,
    masked_nmae,
    masked_rmse,
    ramp_labels,
    skill_score,
)


def test_masked_metrics_match_manual_values() -> None:
    observed = np.array([0.0, 1.0, 9.0])
    predicted = np.array([0.2, 0.6, np.nan])
    mask = np.array([True, True, False])
    assert masked_mae(observed, predicted, mask) == pytest.approx(0.3)
    assert masked_nmae(observed, predicted, mask) == pytest.approx(0.3)
    assert masked_rmse(observed, predicted, mask) == pytest.approx(np.sqrt(0.1))


def test_metrics_reject_all_missing_and_zero_persistence_error() -> None:
    with pytest.raises(ValueError, match="no valid targets"):
        masked_mae(np.array([1.0]), np.array([1.0]), np.array([False]))
    with pytest.raises(ValueError, match="positive"):
        skill_score(0.0, 0.0)


def test_smart_persistence_scales_clear_sky_and_handles_night() -> None:
    np.testing.assert_allclose(smart_persistence(0.4, 400.0, [200.0, 800.0]), [0.2, 0.8])
    np.testing.assert_array_equal(smart_persistence(0.1, 0.0, [0.0, 100.0]), [0.0, 0.0])


def test_climatology_uses_only_rows_passed_to_fit() -> None:
    training = pd.DataFrame(
        {
            "ss_id": [1, 1],
            "datetime_GMT": pd.to_datetime(["2020-01-01T12:00Z", "2020-01-02T12:00Z"]),
            "normalized_power": [0.2, 0.4],
            "is_valid": [True, True],
        }
    )
    future_leak = training.copy()
    future_leak["datetime_GMT"] += pd.Timedelta(days=10)
    future_leak["normalized_power"] = 1.0
    model = SiteTimeClimatology().fit(training)
    assert model.predict("1", ["2020-01-20T12:00Z"])[0] == pytest.approx(0.3)
    assert model.training_end_utc < future_leak["datetime_GMT"].min()


def test_ramp_labels_and_tolerance_matching() -> None:
    assert ramp_labels(np.array([0.2, 0.8, 0.5]), np.array([0.5, 0.5, 0.55])).tolist() == [
        1,
        -1,
        0,
    ]
    times = pd.date_range("2020-01-01", periods=4, freq="15min", tz="UTC")
    score = event_precision_recall_f1(
        times,
        np.array([1, 0, 0, 0]),
        np.array([0, 1, 0, 1]),
        direction=1,
        tolerance_minutes=15,
    )
    assert score == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 0,
        "precision": 0.5,
        "recall": 1.0,
        "f1": pytest.approx(2 / 3),
    }
