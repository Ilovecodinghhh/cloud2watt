import numpy as np
import pytest

from cloud2watt.baselines import normalized_mae, persistence


def test_persistence_repeats_latest_observation() -> None:
    np.testing.assert_array_equal(persistence(0.4, 3), np.array([0.4, 0.4, 0.4]))


def test_persistence_rejects_invalid_horizon() -> None:
    with pytest.raises(ValueError, match="horizons"):
        persistence(0.4, 0)


def test_normalized_mae() -> None:
    assert normalized_mae([0.0, 1.0], [0.2, 0.8], capacity=2.0) == pytest.approx(0.1)


def test_normalized_mae_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same non-empty shape"):
        normalized_mae([0.0], [0.0, 1.0], capacity=1.0)
