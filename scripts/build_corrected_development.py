"""Rebuild existing dates/sites directly into shared tiles with corrected geolocation."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from cloud2watt.data.seviri import SEVIRI_PROJ4, SeviriStore, SitePixel
from cloud2watt.data.shared_tiles import SCHEMA, SharedFrames, intersections, store_bytes
from cloud2watt.run_state import atomic_json, file_hash, json_hash


def build(source, output, cache):
    if output.exists():
        raise FileExistsError(output)
    disk = shutil.disk_usage(output.parent)
    if disk.free - .2 * disk.total < 2_000_000_000:
        raise ValueError("need 2 GB above 20 percent free-space reserve")
    start = time.perf_counter()
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    client = SeviriStore(manifest['source']['satellite'], cache_dir=cache)
    if client.metadata_sha256 != manifest['source']['satellite_metadata_sha256']:
        raise ValueError('source metadata identity changed')
    sites = pd.read_parquet(source/'sites.parquet')
    pixels = client.map_sites(sites)
    old = zarr.open_group(source/'satellite_frames.zarr', mode='r')
    times = np.asarray(old['time_ns'][:])
    source_indices = client.timestamps().get_indexer(pd.to_datetime(times, utc=True))
    if (source_indices < 0).any():
        raise ValueError('historical frame absent from pinned source')
    channels = [client.channels().index(c) for c in manifest['channels']]
    height, width = manifest['crop_shape']
    origins = [(p.y_index-height//2, p.x_index-width//2) for p in pixels]
    y, x = np.min(origins, axis=0).tolist()
    y1, x1 = (np.max(origins, axis=0) + [height, width]).tolist()
    if (y1-y)*(x1-x) > 256**2:
        raise ValueError('bounded compact-region builder refuses a larger region')
    bbox = SitePixel('region', x+(x1-x)//2, y+(y1-y)//2, 0, 0)
    keys = client.required_data_keys(source_indices.tolist(), [bbox], height=y1-y, width=x1-x)
    missing = [k for k in keys if not (cache/k).exists()]
    if len(missing) > 16:
        raise ValueError('more than 16 uncached source chunks; inspect download budget first')
    output.mkdir(parents=True, exist_ok=False)
    root = zarr.open_group(output/'satellite_frames.zarr', mode='w')
    tiles = sorted({(p[0], p[1]) for oy, ox in origins
                    for p in intersections(oy, ox, height, width, 128)})
    root.attrs.update({'schema': SCHEMA, 'complete': False, 'tile_size': 128,
                       'time_chunk': 12, 'site_origins': origins, 'tiles': tiles,
                       'site_ids': sites.ss_id.astype(str).tolist(),
                       'crop_shape': [height, width], 'channel_names': manifest['channels'],
                       'geometry': 'corrected_sweep_y', 'projection_proj4': SEVIRI_PROJ4,
                       'source_metadata_sha256': client.metadata_sha256})
    root.create_array('time_ns', data=times)
    root.create_array('source_frame_indices', data=source_indices)
    arrays, coverage = {}, {}
    for ty, tx in tiles:
        arrays[ty, tx] = root.create_array(f'tiles/{ty}_{tx}',
            shape=(len(times), len(channels), 128, 128), chunks=(12, len(channels), 128, 128),
            dtype='float16', fill_value=np.nan)
        mask = np.zeros((128, 128), dtype=bool)
        for oy, ox in origins:
            for py, px, a, b, c, d in intersections(oy, ox, height, width, 128):
                if (py, px) == (ty, tx):
                    mask[a-ty:b-ty, c-tx:d-tx] = True
        coverage[ty, tx] = mask
        root.create_array(f'coverage/{ty}_{tx}', data=mask)
    for first in range(0, len(times), 12):
        last = min(first+12, len(times))
        block = np.stack([client.read_crop(time_index=int(i), site=bbox,
            channel_indices=channels, height=y1-y, width=x1-x)
            for i in source_indices[first:last]])
        if not np.isfinite(block).all():
            raise ValueError('corrected pixels require fresh QC; refusing to reuse old sample mask')
        for ty, tx, a, b, c, d in intersections(y, x, y1-y, x1-x, 128):
            if (ty, tx) not in arrays:
                continue
            part = np.full((last-first, len(channels), 128, 128), np.nan, dtype='float16')
            part[:, :, a-ty:b-ty, c-tx:d-tx] = block[:, :, a-y:b-y, c-x:d-x]
            part[:, :, ~coverage[ty, tx]] = np.nan
            arrays[ty, tx][first:last] = part
        if first % 480 == 0:
            print(f'corrected frames {last}/{len(times)}', flush=True)
    root.attrs['complete'] = True
    reader = SharedFrames(root)
    for site in range(len(sites)):
        for slot in (0, len(times)//2, len(times)-1):
            expected = client.read_crop(time_index=int(source_indices[slot]), site=pixels[site],
                                        channel_indices=channels, height=height, width=width)
            np.testing.assert_array_equal(reader[site, slot], expected)
    for name in ('power_15min.parquet', 'sample_index.parquet'):
        shutil.copyfile(source/name, output/name)
    sites['satellite_x_index'] = [p.x_index for p in pixels]
    sites['satellite_y_index'] = [p.y_index for p in pixels]
    sites.to_parquet(output/'sites.parquet', index=False)
    manifest.pop('manifest_content_sha256')
    manifest.update({'dataset_version': output.name, 'schema_version': 3,
                     'parent_manifest_sha256': file_hash(source/'manifest.json'),
                     'power_value_semantics': 'mean_of_three_scaled_instantaneous_samples',
                     'label_contract': 'c2w-sampled-power-v1',
                     'evaluation_protocol': 'c2w-eval-v2.1',
                     'satellite_layout': SCHEMA, 'projection_proj4': SEVIRI_PROJ4,
                     'availability_basis': 'idealized_archive_time',
                     'data_role': 'historical_development_only'})
    manifest['manifest_content_sha256'] = json_hash(manifest)
    atomic_json(output/'manifest.json', manifest)
    report = {'source': str(source), 'output': str(output), 'corrected_frames': len(times),
              'site_count': len(sites), 'direct_source_crop_checks': len(sites)*3,
              'tiles': len(tiles), 'all_corrected_bbox_pixels_finite': True,
              'power_and_sample_files_unchanged': all(file_hash(source/n) == file_hash(output/n)
                  for n in ('power_15min.parquet', 'sample_index.parquet')),
              'downloaded_bytes': client.downloaded_bytes, 'uncached_source_chunks': len(missing),
              'elapsed_seconds': time.perf_counter()-start, 'derived_bytes': store_bytes(output)}
    atomic_json(output/'build_report.json', report)
    atomic_json(output/'complete.json', {
        'manifest': manifest['manifest_content_sha256'],
        'build_report_sha256': file_hash(output/'build_report.json')})
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/processed/paired-30d-v1'))
    parser.add_argument('--output', type=Path,
                        default=Path('data/processed/paired-30d-corrected-v2'))
    parser.add_argument('--cache', type=Path, default=Path('outputs/cache/seviri-2020'))
    args = parser.parse_args()
    build(args.source, args.output, args.cache)
