"""Frozen small-budget development comparison and offline replay demo."""
from __future__ import annotations

import argparse
import gc
import html
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cloud2watt.baselines import smart_persistence
from cloud2watt.evaluation.forecast import SELECTION_METRIC, score_forecast
from cloud2watt.evaluation.splits import build_split_manifest
from cloud2watt.experiment import execute, parser_for, prepare
from cloud2watt.experiment_matrix import rank_completed_runs
from cloud2watt.run_state import atomic_json, atomic_path, file_hash, is_complete
from cloud2watt.training import FeatureStatistics, PowerForecastDataset


def reference_scores(data, output):
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    samples = pd.read_parquet(data/'sample_index.parquet')
    sites = pd.read_parquet(data/'sites.parquet')
    power = pd.read_parquet(data/'power_15min.parquet')
    assigned, _ = build_split_manifest(samples, sites,
        data_manifest_hash=manifest['manifest_content_sha256'], seed=42, embargo_hours=4)
    train = assigned.loc[(assigned.temporal_split == 'train') &
                         (assigned.site_split == 'development')]
    stats = FeatureStatistics.fit(train, power, sites)
    results, frames = {}, []
    for group, label in [('development', 'validation_development'),
                         ('holdout', 'validation_holdout')]:
        rows = assigned.loc[(assigned.temporal_split == 'validation') &
                            (assigned.site_split == group)].reset_index(drop=True)
        dataset = PowerForecastDataset(rows, power, sites, stats)
        target = np.stack([r['target'] for r in dataset.records])
        mask = np.stack([r['target_mask'] for r in dataset.records])
        solar = np.stack([r['target_solar_elevation_deg'] for r in dataset.records])
        current = power.iloc[[int(x[-1]) for x in rows.power_history_row_indices]]
        indices = np.stack(rows.target_row_indices).astype(int)
        future = power.clear_sky_ghi_wm2.to_numpy()[np.maximum(indices, 0)]
        predictions = {
            'persistence': np.repeat(current.normalized_power.to_numpy()[:, None], 6, axis=1),
            'smart_persistence': np.stack([smart_persistence(p, g, f) for p, g, f in zip(
                current.normalized_power, current.clear_sky_ghi_wm2, future, strict=True)])}
        results[label] = {}
        for name, prediction in predictions.items():
            results[label][name] = score_forecast(prediction, target, mask, solar, rows)
            frames.append(rows[['site_id', 'issue_time_utc']].assign(
                prediction=list(prediction), observed=list(target), target_mask=list(mask),
                partition=label, model=name))
    atomic_json(output/'baselines.json', results)
    pd.concat(frames).to_parquet(output/'baseline_predictions.parquet', index=False)
    return results


def summarize(paths, references, output):
    ranking = rank_completed_runs(paths)
    rows = []
    for candidate in ranking['candidates']:
        runs = candidate['runs']
        scores = [r['score'] for r in runs]
        holdout, epochs = [], []
        for run in runs:
            folder = Path(run['path'])
            metrics = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
            if metrics['validation_development']['none']['comparison'] != references[
                    'validation_development']['persistence']['comparison']:
                raise ValueError('baseline and learning candidate populations differ')
            holdout.append(metrics['validation_holdout']['none'][SELECTION_METRIC])
            epochs.append(torch.load(folder/'best.pt', weights_only=True)['epoch'])
        rows.append({'model': candidate['mode'], 'parameters': candidate['parameters'],
                     'development_mean': float(np.mean(scores)),
                     'seed_sd': float(np.std(scores, ddof=1)),
                     'holdout_mean': float(np.mean(holdout)), 'best_epochs': epochs,
                     'selection_ready': candidate['selection_ready'], 'runs': runs})
    for name, result in references['validation_development'].items():
        rows.append({'model': name, 'parameters': 0, 'development_mean': result[SELECTION_METRIC],
                     'seed_sd': None, 'holdout_mean': references['validation_holdout'][name][
                         SELECTION_METRIC], 'best_epochs': [], 'selection_ready': True, 'runs': []})
    rows.sort(key=lambda row: row['development_mean'])
    result = {'protocol': ranking['protocol'], 'label_contract': 'c2w-sampled-power-v1',
              'scope': 'winter historical development, fixed eight-epoch budget',
              'rows': rows, 'ranking': ranking}
    atomic_json(output/'comparison.json', result)
    return result


def make_demo(result, output):
    frames = [pd.read_parquet(output/'baseline_predictions.parquet')]
    for row in result['rows']:
        for run in row['runs']:
            if run['seed'] == 42:
                f = pd.read_parquet(Path(run['path'])/'predictions.parquet')
                frames.append(f.loc[f.perturbation.eq('none')].assign(model=row['model']))
    records = {}
    for frame in frames:
        for row in frame.itertuples():
            if row.partition != 'validation_development':
                continue
            key = (str(row.site_id), str(row.issue_time_utc))
            if key not in records:
                records[key] = {'site': key[0], 'time': key[1],
                    'observed': [float(v) if m else None for v, m in zip(
                        row.observed, row.target_mask, strict=True)], 'predictions': {}}
            records[key]['predictions'][row.model] = [float(v) for v in row.prediction]
    payload = json.dumps(list(records.values()), allow_nan=False).replace('</', '<\\/')
    table = ''.join('<tr><td>'+html.escape(r['model'])+'</td><td>'
                    +f'{r["development_mean"]*100:.3f}'
                    +'</td><td>'+('—' if r['seed_sd'] is None else f'{r["seed_sd"]*100:.3f}')
                    +'</td><td>'+f'{r["holdout_mean"]*100:.3f}'+'</td></tr>'
                    for r in result['rows'])
    template = (Path(__file__).resolve().parents[1]/'demo'/'template.html').read_text('utf-8')
    demo = output/'demo.html'
    with atomic_path(demo) as temp:
        temp.write_text(template.replace('__TABLE__', table).replace('__DATA__', payload),
                        encoding='utf-8')
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path,
                        default=Path('data/processed/paired-30d-corrected-v2'))
    parser.add_argument('--output', type=Path, default=Path('outputs/compact-route-v21'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    references = reference_scores(args.data, args.output)
    paths, timings = [], []
    for kind, mode in [('power', 'mlp'), ('satellite', 'power_solar'), ('satellite', 'full')]:
        config = Path(f'configs/compact_{"power" if kind == "power" else "satellite"}.yaml')
        for seed in (42, 123, 2026):
            options = ['--data', str(args.data), '--config', str(config), '--output-root',
                       str(args.output/'runs'), '--seed', str(seed), '--auto-resume',
                       '--model' if kind == 'power' else '--mode', mode]
            run_args = parser_for(kind).parse_args(options)
            started = time.perf_counter()
            prepared = prepare(kind, run_args)
            path = execute(prepared, run_args)
            if not is_complete(path, prepared['identity']):
                raise ValueError('matrix run incomplete')
            paths.append(path)
            timings.append({'mode': mode, 'seed': seed, 'seconds': time.perf_counter()-started,
                            'run': str(path), 'complete_sha256': file_hash(path/'complete.json')})
            atomic_json(args.output/'progress.json', timings)
            del prepared
            gc.collect()
    result = summarize(paths, references, args.output)
    demo = make_demo(result, args.output)
    print(json.dumps({'comparison': str(args.output/'comparison.json'),
                      'demo': str(demo)}, indent=2))


if __name__ == '__main__':
    main()
