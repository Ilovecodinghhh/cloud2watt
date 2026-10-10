"""Bounded migration and matched loader benchmark; no training or test scoring."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import DataLoader

from cloud2watt.data.satellite_loader import (
    ChunkBucketSampler,
    SatelliteForecastDataset,
    fit_satellite_statistics,
)
from cloud2watt.data.shared_tiles import (
    migrate_site_frames,
    open_satellite_frames,
    store_bytes,
    write_json,
)
from cloud2watt.training import FeatureStatistics


def rss_bytes(pid):
    """Read OS working set, including independently queried loader workers."""
    if os.name != 'nt':
        return int(Path(f'/proc/{pid}/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                'peak_rss', 'rss', 'peak_pool_paged', 'pool_paged', 'peak_pool_nonpaged',
                'pool_nonpaged', 'pagefile', 'peak_pagefile')]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    handle = kernel.OpenProcess(0x1000 | 0x10, False, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), 'OpenProcess failed')
    try:
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), 'GetProcessMemoryInfo failed')
        return int(counters.rss)
    finally:
        kernel.CloseHandle(handle)


def benchmark(dataset, samples, *, batches, workers):
    loader = DataLoader(dataset, batch_size=8, sampler=ChunkBucketSampler(samples),
                        num_workers=workers, persistent_workers=workers > 0,
                        **({'prefetch_factor': 2} if workers else {}))
    started = time.perf_counter()
    iterator = iter(loader)
    worker_pids = [w.pid for w in iterator._workers] if workers else []
    latencies, curve, read = [], [], 0
    for step in range(batches):
        tick = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        latencies.append(time.perf_counter()-tick)
        read += len(batch['satellite'])
        if step % 25 == 0 or step == batches-1:
            curve.append({'batch': step+1, 'elapsed_s': time.perf_counter()-started,
                          'parent_rss_bytes': rss_bytes(os.getpid()),
                          'worker_rss_bytes': [rss_bytes(pid) for pid in worker_pids]})
    elapsed = time.perf_counter()-started
    # A second pass reuses workers and decoded caches; OS cache is not evicted.
    warm_started, warm_latencies, warm_read = time.perf_counter(), [], 0
    iterator = iter(loader)
    for _ in range(min(100, len(loader))):
        tick = time.perf_counter()
        batch = next(iterator)
        warm_latencies.append(time.perf_counter()-tick)
        warm_read += len(batch['satellite'])
    warm_elapsed = time.perf_counter()-warm_started
    if workers:
        iterator._shutdown_workers()
    tail = curve[len(curve)//2:]
    totals = [r['parent_rss_bytes'] + sum(r['worker_rss_bytes']) for r in tail]
    result = {
        'batches': batches, 'samples': read, 'workers': workers, 'batch_size': 8,
        'first_pass_seconds': elapsed, 'first_pass_samples_per_second': read/elapsed,
        'first_batch_seconds': latencies[0],
        'p95_batch_seconds': float(np.quantile(latencies, .95)),
        'warm_samples_per_second': warm_read/warm_elapsed,
        'warm_p95_batch_seconds': float(np.quantile(warm_latencies, .95)),
        'rss_curve': curve, 'post_warmup_rss_range_bytes': max(totals)-min(totals),
        'post_warmup_rss_slope_bytes_per_batch': float(np.polyfit(
            [r['batch'] for r in tail], totals, 1)[0]) if len(tail) > 1 else None,
        'cache_note': 'fresh loader first pass; OS cache not evicted; second pass warm',
        'unique_dataset_samples': len(dataset), 'repeated_epochs': read > len(dataset),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/processed/paired-30d-v1'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=192)
    parser.add_argument('--batches', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.frames < 8 or args.batches < 50 or args.workers < 0:
        parser.error('require frames >= 8, batches >= 50, workers >= 0')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    sites = pd.read_parquet(args.data/'sites.parquet')
    samples = pd.read_parquet(args.data/'sample_index.parquet')
    power = pd.read_parquet(args.data/'power_15min.parquet')
    old = args.data/'satellite_frames.zarr'
    old_root = zarr.open_group(old, mode='r')
    if args.frames > old_root['frames'].shape[1]:
        parser.error('requested more frames than source contains')
    samples = samples.loc[samples.satellite_frame_indices.map(
        lambda values: max(values) < args.frames)].reset_index(drop=True)
    if samples.empty:
        raise ValueError('selected window has no samples')
    shared_path = args.output/'shared.zarr'
    started = time.perf_counter()
    migration = migrate_site_frames(old, shared_path, sites, frame_indices=np.arange(args.frames))
    migration['seconds'] = time.perf_counter()-started
    migration['shared_bytes'] = store_bytes(shared_path)
    migration['source_window_chunk_bytes'] = sum(
        p.stat().st_size for p in (old/'frames'/'c').rglob('*')
        if p.is_file() and int(p.relative_to(old/'frames'/'c').parts[1]) < (args.frames+11)//12)
    shared = open_satellite_frames(shared_path)
    checked = 0
    for site in range(len(sites)):
        for start in range(0, args.frames, 12):
            indices = np.arange(start, min(start+12, args.frames))
            np.testing.assert_array_equal(shared[site, indices], old_root['frames'][site, indices])
            checked += len(indices)
    migration['exact_frames_verified'] = checked
    write_json(args.output/'migration.json', migration)
    feature_stats = FeatureStatistics.fit(samples, power, sites)
    satellite_stats = fit_satellite_statistics(samples, old, maximum_frames=64)
    new_stats = fit_satellite_statistics(samples, shared_path, maximum_frames=64)
    assert satellite_stats == new_stats
    results = {}
    for name, path in [('legacy', old), ('shared', shared_path)]:
        dataset = SatelliteForecastDataset(
            samples, power, sites, feature_stats, satellite_stats, path)
        results[name] = benchmark(dataset, samples, batches=args.batches, workers=args.workers)
        write_json(args.output/f'{name}-benchmark.json', results[name])
        print(name, results[name]['warm_samples_per_second'], flush=True)
    disk = shutil.disk_usage(args.output)
    report = {'scope': 'two-day historical development storage benchmark; no scoring',
              'migration': migration, 'loaders': results,
              'network_bytes_this_migration': 0, 'temporary_duplicate_store_bytes': 0,
              'disk_total_bytes': disk.total, 'disk_free_bytes': disk.free,
              'source_cache_bytes': store_bytes(Path('outputs/cache/seviri-2020')),
              'torch': str(torch.__version__), 'statistics_equal': True}
    write_json(args.output/'report.json', report)
    print(json.dumps({'migration': migration, 'disk_free_bytes': disk.free}, indent=2))


if __name__ == '__main__':
    main()
