#!/usr/bin/env python3
"""Combine disjoint sequential-ablation sweeps after checking their provenance."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from analysis.expanded.methods import METHODS
from analysis.large_study.evaluate import FRACTIONS, METRICS, paired_comparisons, summarize, write_json
from scripts.ablate import file_digest

RESULTS = ROOT / 'results/large_study'


def combine_rows(initial, additional):
    """Join the interior fractions, keeping the initial run's endpoints."""
    keys = ['window_id', 'cluster_id', 'method', 'fraction']
    for frame in (initial, additional):
        if frame[keys].duplicated().any() or not np.isfinite(frame[list(METRICS)].to_numpy()).all():
            raise ValueError('Duplicate or non-finite measurements')
    if set(initial.window_id) != set(additional.window_id):
        raise ValueError('The sweeps use different windows')
    for frame in (initial, additional):
        if not frame.groupby('window_id').cluster_id.nunique().eq(1).all():
            raise ValueError('A window has inconsistent article identifiers')
    initial_articles = initial.groupby('window_id').cluster_id.first()
    additional_articles = additional.groupby('window_id').cluster_id.first()
    if not initial_articles.equals(additional_articles):
        raise ValueError('The sweeps assign windows to different articles')
    left = initial[initial.method.isin(['identity', 'all_zero'])].sort_values(keys)
    right = additional[additional.method.isin(['identity', 'all_zero'])].sort_values(keys)
    if not left[keys].reset_index(drop=True).equals(right[keys].reset_index(drop=True)):
        raise ValueError('Endpoint windows differ')
    if not np.allclose(left[list(METRICS)], right[list(METRICS)], rtol=1e-5, atol=1e-6):
        raise ValueError('Endpoint measurements differ')
    extra = additional[additional.fraction.isin(FRACTIONS)]
    result = pd.concat([initial, extra], ignore_index=True)
    expected = {(window, method, fraction) for window in initial.window_id.unique()
                for method in (*METHODS, 'random') for fraction in (*FRACTIONS, 1.)}
    expected |= {(window, method, fraction) for window in initial.window_id.unique()
                 for method, fraction in [('identity', 0.), ('all_zero', 1.)]}
    if len(result) != len(expected) or set(zip(result.window_id, result.method, result.fraction)) != expected:
        raise ValueError('The combined sweep does not contain exactly the full fraction grid')
    return result


def combine(model):
    directories = [RESULTS / name / model for name in ('all_layers', 'all_layers_additional')]
    runs, signatures = [], []
    for directory in directories:
        run = json.loads((directory / 'run.json').read_text())
        signature = json.loads((directory / 'signature.json').read_text())
        if run['status'] != 'passed' or run['signature_sha256'] != file_digest(directory / 'signature.json'):
            raise ValueError('Source run is incomplete or its signature changed')
        runs.append(run)
        signatures.append(signature)
    for field in ('model', 'model_commit', 'device', 'batch_size', 'manifest', 'source_sha256'):
        if signatures[0][field] != signatures[1][field]:
            raise ValueError(f'Sweep provenance differs: {field}')
    if signatures[0]['model'] != model or any(runs[0][field] != runs[1][field] for field in ('layers', 'windows', 'articles')):
        raise ValueError('Model or sample counts differ')
    for layer in range(runs[0]['layers']):
        paths = [directories[0] / f'calibration_{layer}.pt', directories[1] / f'calibration_{layer}.pt']
        if not paths[1].exists():
            paths[1] = paths[0]
        for path, signature in zip(paths, signatures):
            if file_digest(path) != signature['calibration_sha256'][layer]:
                raise ValueError(f'Calibration hash differs at layer {layer}')
        first, second = [torch.load(path, map_location='cpu', weights_only=False) for path in paths]
        if first['activation_sha256'] != second['activation_sha256'] or not torch.equal(first['means'], second['means']):
            raise ValueError(f'Training calibration differs at layer {layer}')
    frames = [pd.read_parquet(directory / 'metrics.parquet') for directory in directories]
    combined = combine_rows(*frames)
    destination = RESULTS / 'all_layers_combined' / model
    destination.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(destination / 'metrics.parquet', index=False)
    summarize(combined.to_dict('records')).to_csv(destination / 'summary.csv', index=False)
    paired_comparisons(combined.to_dict('records')).to_csv(destination / 'paired_comparisons.csv', index=False)
    write_json(destination / 'run.json', {'status': 'passed', 'windows': runs[0]['windows'],
               'articles': runs[0]['articles'], 'layers': runs[0]['layers'], 'fractions': list(FRACTIONS),
               'sources': [{'directory': str(path.relative_to(ROOT)), 'run_sha256': file_digest(path / 'run.json'),
                            'signature_sha256': file_digest(path / 'signature.json'),
                            'metrics_sha256': file_digest(path / 'metrics.parquet')} for path in directories],
               'combiner_sha256': file_digest(Path(__file__))})
    print(f'{model}: combined {len(combined)} window measurements', flush=True)


if __name__ == '__main__':
    for model in ('gpt2', 'smollm2'):
        combine(model)
