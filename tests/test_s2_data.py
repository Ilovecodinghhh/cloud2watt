import pickle

import numpy as np
import pandas as pd
import pytest
import zarr
from pyproj import CRS, Transformer

from cloud2watt.data.availability import availability_contract, delayed_history_indices
from cloud2watt.data.seviri import SEVIRI_PROJ4
from cloud2watt.data.shared_tiles import SharedFrames, migrate_site_frames


def fixture_store(tmp_path):
    # Overlap across tile boundaries, reverse channels and one missing frame.
    source = np.arange(8*2*16*16, dtype=np.float32).reshape(8, 2, 16, 16)
    source[3] = np.nan
    sites = pd.DataFrame({'ss_id': ['a', 'b'], 'satellite_y_index': [6, 8],
                          'satellite_x_index': [6, 8]})
    root = zarr.open_group(tmp_path / 'old', mode='w')
    frames = np.stack([source[:, ::-1, 2:10, 2:10], source[:, ::-1, 4:12, 4:12]])
    root.create_array('frames', data=frames, chunks=(1, 3, 2, 8, 8))
    root.create_array('time_ns', data=np.arange(8, dtype=np.int64))
    root.attrs.update({'site_ids': ['a', 'b'], 'channel_names': ['IR_108', 'VIS006']})
    return sites, frames


def test_shared_tiles_exact_overlap_boundary_missing_and_halo(tmp_path):
    sites, frames = fixture_store(tmp_path)
    report = migrate_site_frames(tmp_path/'old', tmp_path/'new', sites,
                                 tile_size=4, time_chunk=3)
    shared = SharedFrames(zarr.open_group(tmp_path/'new', mode='r'), cache_blocks=2)
    for site in range(2):
        np.testing.assert_array_equal(shared[site, [7, 3, 0, 4]], frames[site, [7, 3, 0, 4]])
    assert report['covered_pixels'] == 92
    assert len(shared.cache) <= 2
    np.testing.assert_array_equal(shared[0, 1], frames[0, 1])
    with pytest.raises(ValueError, match='coverage'):
        shared.read(0, [1], halo=1)
    with pytest.raises(IndexError):
        shared[0, [-1]]
    with pytest.raises(FileExistsError):
        migrate_site_frames(tmp_path/'old', tmp_path/'new', sites)
    # Reopening remains independent of live cache state.
    assert pickle.loads(pickle.dumps(shared)).shape == shared.shape


def test_conflicting_overlap_never_marks_store_complete(tmp_path):
    sites, _ = fixture_store(tmp_path)
    root = zarr.open_group(tmp_path/'old', mode='a')
    root['frames'][1, 0, 0, 0, 0] = -99
    with pytest.raises(ValueError, match='disagree'):
        migrate_site_frames(tmp_path/'old', tmp_path/'new', sites, tile_size=4)
    with pytest.raises(ValueError, match='incomplete'):
        SharedFrames(zarr.open_group(tmp_path/'new', mode='r'))


@pytest.mark.parametrize('delay', [0, 15, 30])
def test_availability_never_selects_future_or_fills_missing(delay):
    times = pd.date_range('2020-12-01', periods=12, freq='15min', tz='UTC').delete(4)
    ledger = availability_contract(times, delay_minutes=delay)
    issue = pd.DatetimeIndex([pd.Timestamp('2020-12-01T02:00Z')])
    indices = delayed_history_indices(issue, ledger, delay_minutes=delay)[0]
    valid = indices >= 0
    assert (ledger.available_time.iloc[indices[valid]] <= issue[0]).all()
    if delay:
        assert not valid.all()
    with pytest.raises(ValueError, match='publication'):
        availability_contract(times, delay_minutes=delay, basis='measured')


def test_meteosat_projection_uses_y_sweep():
    assert '+sweep=y' in SEVIRI_PROJ4
    actual = Transformer.from_crs('EPSG:4326', CRS.from_proj4(SEVIRI_PROJ4), always_xy=True)
    metadata = {'proj': 'geos', 'lon_0': 9.5, 'h': 35785831, 'a': 6378169,
                'rf': 295.488065897014}
    expected = Transformer.from_crs('EPSG:4326', CRS.from_dict(metadata), always_xy=True)
    np.testing.assert_allclose(actual.transform(-1.79, 51.6), expected.transform(-1.79, 51.6))


def test_halo_assembled_from_neighbor_tiles_when_covered(tmp_path):
    sites, _ = fixture_store(tmp_path)
    migrate_site_frames(tmp_path/'old', tmp_path/'new', sites, tile_size=4, time_chunk=3)
    root = zarr.open_group(tmp_path/'new', mode='a')
    # Reuse a covered inner crop to exercise halo crossing two tile boundaries.
    root.attrs['site_origins'] = [[4, 4]]
    root.attrs['site_ids'] = ['inner']
    root.attrs['crop_shape'] = [4, 4]
    shared = SharedFrames(root)
    old = zarr.open_group(tmp_path/'old', mode='r')['frames']
    np.testing.assert_array_equal(shared.read(0, [0, 3, 7], halo=1),
                                  old[0, [0, 3, 7]][:, :, 1:7, 1:7])


def test_shared_loader_preserves_model_inputs_and_targets(tmp_path):
    from cloud2watt.data.satellite_loader import (
        SatelliteForecastDataset,
        fit_satellite_statistics,
    )
    from cloud2watt.training import FeatureStatistics

    sites, _ = fixture_store(tmp_path)
    sites = sites.assign(kWp=4., tilt=30., orientation=180.,
                         latitude_rounded=51., longitude_rounded=-1.)
    times = pd.date_range('2020-12-01', periods=8, freq='15min', tz='UTC')
    root = zarr.open_group(tmp_path/'old', mode='a')
    root['time_ns'][:] = times.to_numpy(dtype='datetime64[ns]').astype('int64')
    migrate_site_frames(tmp_path/'old', tmp_path/'new', sites, tile_size=4, time_chunk=3)
    power = pd.DataFrame({'normalized_power': np.linspace(.1, .5, 8),
                          'solar_elevation_deg': 20., 'solar_azimuth_deg': 150.,
                          'clear_sky_ghi_wm2': 500.})
    samples = pd.DataFrame({'site_id': ['a'], 'site_index': [0],
                            'issue_time_utc': [times[3]],
                            'satellite_frame_indices': [np.arange(4)],
                            'power_history_row_indices': [np.arange(4)],
                            'target_row_indices': [np.array([4, 5, 6, 7, -1, -1])],
                            'target_mask': [[True, True, True, True, False, False]]})
    fs = FeatureStatistics.fit(samples, power, sites)
    stats = fit_satellite_statistics(samples, tmp_path/'old')
    assert stats == fit_satellite_statistics(samples, tmp_path/'new')
    old = SatelliteForecastDataset(samples, power, sites, fs, stats, tmp_path/'old')
    new = SatelliteForecastDataset(samples, power, sites, fs, stats, tmp_path/'new')
    for key, value in old[0].items():
        if hasattr(value, 'numpy'):
            np.testing.assert_array_equal(value.numpy(), new[0][key].numpy())
        else:
            assert value == new[0][key]
    assert new[0]['satellite_mask'].tolist() == [True, True, True, False]
