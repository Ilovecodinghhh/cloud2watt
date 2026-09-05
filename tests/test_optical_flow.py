import numpy as np
import pytest

from cloud2watt.optical_flow import (
    MaskedRidge,
    cached_flow_feature_matrix,
    flow_sequence_features,
    phase_correlation_translation,
)


def test_phase_correlation_recovers_synthetic_translation() -> None:
    rng = np.random.default_rng(42)
    first = rng.normal(size=(64, 64))
    second = np.roll(first, shift=(8, -4), axis=(0, 1))
    dy, dx = phase_correlation_translation(first, second, downsample=1)
    assert dy == pytest.approx(8)
    assert dx == pytest.approx(-4)


def test_flow_features_have_three_pair_records() -> None:
    first = np.zeros((32, 32))
    first[8:16, 10:18] = 1
    history = np.stack([np.roll(first, shift=(index, 0), axis=(0, 1)) for index in range(4)])
    assert flow_sequence_features(history, downsample=1).shape == (15,)


def test_masked_ridge_fits_each_horizon_without_invalid_targets() -> None:
    features = np.arange(20, dtype=float).reshape(10, 2)
    targets = np.column_stack((features[:, 0] * 2, features[:, 1] * -1))
    mask = np.ones_like(targets, dtype=bool)
    targets[-1, 1] = 1_000_000
    mask[-1, 1] = False
    model = MaskedRidge(alpha=0.0).fit(features, targets, mask)
    prediction = model.predict(features)
    np.testing.assert_allclose(prediction[:-1], targets[:-1], atol=1e-7)


def test_ridge_requires_fit_before_prediction() -> None:
    with pytest.raises(ValueError, match="fitted"):
        MaskedRidge().predict(np.zeros((1, 2)))


def test_flow_cache_is_reused(tmp_path) -> None:
    import pandas as pd
    import zarr

    root = zarr.open_group(tmp_path / "frames.zarr", mode="w")
    frames = np.zeros((1, 4, 3, 16, 16), dtype=np.float32)
    frames[0, :, 2, 4:8, 4:8] = 1
    root.create_array("frames", data=frames)
    samples = pd.DataFrame({"site_index": [0], "satellite_frame_indices": [np.arange(4)]})
    first, cache_key, first_hit = cached_flow_feature_matrix(
        samples, str(tmp_path / "frames.zarr"), tmp_path / "cache", downsample=1
    )
    second, second_key, second_hit = cached_flow_feature_matrix(
        samples, str(tmp_path / "frames.zarr"), tmp_path / "cache", downsample=1
    )
    np.testing.assert_array_equal(first, second)
    assert cache_key == second_key
    assert not first_hit and second_hit
