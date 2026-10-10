"""Audit fixed-source semantics, historical projection, and availability scenarios."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from pyproj import CRS, Geod, Transformer

from cloud2watt.data.alignment import aggregate_power_15min
from cloud2watt.data.availability import availability_contract, delayed_history_indices
from cloud2watt.data.seviri import SEVIRI_PROJ4, SeviriStore, SitePixel
from cloud2watt.data.shared_tiles import write_json
from cloud2watt.data.spatial import nearest_index
from cloud2watt.run_state import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/processed/paired-30d-v1'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.data/'manifest.json').read_text(encoding='utf-8'))
    revision = manifest['source']['pv_revision']
    card = {'revision': revision, 'url':
            f'https://huggingface.co/datasets/openclimatefix/uk_pv/blob/{revision}/README.md'}
    try:
        path = Path(hf_hub_download('openclimatefix/uk_pv', 'README.md',
                                   repo_type='dataset', revision=revision))
        card.update({'status': 'retrieved', 'sha256': file_hash(path),
                     'contains_instantaneous': 'instantaneous' in path.read_text('utf-8'),
                     'contains_multiply_by_12': 'by 12' in path.read_text('utf-8')})
        (args.output/'source-card.md').write_bytes(path.read_bytes())
    except Exception as exc:
        card.update({'status': 'unavailable', 'error_type': type(exc).__name__})
    probe = json.loads(Path('outputs/probes/uk_pv.json').read_text(encoding='utf-8'))
    raw_path = Path(probe['details']['partition']['path'])
    sites = pd.read_parquet(args.data/'sites.parquet')
    power = pd.read_parquet(args.data/'power_15min.parquet')
    raw = pd.read_parquet(raw_path, filters=[
        ('ss_id', 'in', sites.ss_id.tolist()),
        ('datetime_GMT', '>=', pd.Timestamp('2020-12-01T00:00Z')),
        ('datetime_GMT', '<=', pd.Timestamp('2020-12-03T00:00Z'))])
    aligned = aggregate_power_15min(raw, sites.set_index('ss_id').kWp,
                                    value_semantics='interval_energy')
    joined = aligned.merge(power[['ss_id', 'datetime_GMT', 'normalized_power']],
                            on=['ss_id', 'datetime_GMT'], suffixes=('_rebuilt', '_stored'))
    valid = joined[['normalized_power_rebuilt', 'normalized_power_stored']].notna().all(axis=1)
    delta = (joined.loc[valid, 'normalized_power_rebuilt'] -
             joined.loc[valid, 'normalized_power_stored']).abs()
    pv = {'raw_sha256': file_hash(raw_path), 'raw_schema': str(pq.ParquetFile(raw_path).schema),
          'raw_rows_checked': len(raw), 'aligned_bins_compared': int(valid.sum()),
          'max_abs_rebuild_error': float(delta.max()),
          'normalized_power_quantiles': aligned.normalized_power.quantile(
              [0, .5, .95, .99, 1]).to_dict(),
          'normalization': 'mean(12 * generation_Wh) / (kWp * 1000), three valid readings',
          'integrated_30min_comparison': 'not available in existing local snapshot',
          'label_semantics_status': '5-minute instantaneous versus interval support unresolved'}
    client = SeviriStore(manifest['source']['satellite'],
                         cache_dir=Path('outputs/cache/seviri-2020'))
    assert client.metadata_sha256 == manifest['source']['satellite_metadata_sha256']
    x, y = client.coordinate('x_geostationary'), client.coordinate('y_geostationary')
    correct = Transformer.from_crs('EPSG:4326', CRS.from_proj4(SEVIRI_PROJ4), always_xy=True)
    inverse = Transformer.from_crs(CRS.from_proj4(SEVIRI_PROJ4), 'EPSG:4326', always_xy=True)
    geod = Geod(ellps='WGS84')
    geometry = []
    import zarr
    old = zarr.open_group(args.data/'satellite_frames.zarr', mode='r')
    times = pd.to_datetime(old['time_ns'][:], utc=True)
    source_times = client.timestamps()
    source_index = source_times.get_indexer(times)
    channels = [client.channels().index(c) for c in manifest['channels']]
    for index, site in enumerate(sites.iloc[:5].itertuples()):
        px, py = correct.transform(site.longitude_rounded, site.latitude_rounded)
        col, row = nearest_index(x, px), nearest_index(y, py)
        lon, lat = inverse.transform(x[col], y[row])
        lonx, latx = inverse.transform(x[col+1], y[row])
        lony, laty = inverse.transform(x[col], y[row+1])
        old_col, old_row = int(site.satellite_x_index), int(site.satellite_y_index)
        old_lon, old_lat = inverse.transform(x[old_col], y[old_row])
        old_pixel = SitePixel(str(site.ss_id), old_col, old_row, 0, 0)
        crop = client.read_crop(time_index=int(source_index[50]), site=old_pixel,
                               channel_indices=channels, height=128, width=128)
        np.testing.assert_array_equal(crop, old['frames'][index, 50])
        geometry.append({'site_id': str(site.ss_id), 'old_pixel_xy': [old_col, old_row],
                         'correct_pixel_xy': [col, row],
                         'old_center_distance_m': geod.inv(site.longitude_rounded,
                             site.latitude_rounded, old_lon, old_lat)[2],
                         'correct_center_distance_m': geod.inv(site.longitude_rounded,
                             site.latitude_rounded, lon, lat)[2],
                         'local_ground_pixel_x_m': geod.inv(lon, lat, lonx, latx)[2],
                         'local_ground_pixel_y_m': geod.inv(lon, lat, lony, laty)[2],
                         'legacy_source_crop_exact': True})
    samples = pd.read_parquet(args.data/'sample_index.parquet')
    # Audit development-window coverage only; no label scores or final-test selection.
    issues = pd.DatetimeIndex(sorted(samples.loc[
        samples.issue_time_utc < pd.Timestamp('2020-12-03T00:00Z'), 'issue_time_utc'].unique()))
    availability = []
    for delay in (0, 15, 30):
        ledger = availability_contract(times, delay_minutes=delay)
        selected = delayed_history_indices(issues, ledger, delay_minutes=delay)
        ledger.to_parquet(args.output/f'availability-{delay}min.parquet', index=False)
        availability.append({'satellite_delay_minutes': delay, 'basis': 'scenario',
                             'issue_times': len(issues),
                             'complete_histories': int((selected >= 0).all(axis=1).sum()),
                             'missing_slots': int((selected < 0).sum()),
                             'scores': 'not evaluated; this is input-availability replay only'})
    attrs = client.metadata['data/.zattrs']
    selected_attrs = {k: v for k, v in attrs.items() if k.startswith(('VIS006_', 'IR_108_'))
                      and any(k.endswith(t) for t in ('units', 'start_time', 'end_time', 'area'))}
    report = {'pv_card': card, 'pv_audit': pv, 'projection': SEVIRI_PROJ4,
              'site_checks': geometry, 'source_attributes': selected_attrs,
              'satellite_metadata_sha256': client.metadata_sha256,
              'availability_scenarios': availability,
              'source_network_bytes_this_audit': client.downloaded_bytes,
              'raw_time_coordinate_hash': hashlib.sha256(old['time_ns'][:].tobytes()).hexdigest(),
              'satellite_time_semantics': 'archive time coordinate; per-pixel scan and publication '
              'timestamps not provided; annual attributes do not establish per-frame availability',
              'large_build_allowed': False}
    write_json(args.output/'audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
