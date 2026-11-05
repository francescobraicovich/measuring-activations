#!/usr/bin/env python3
"""Exploratory position-localized zero ablations and local projection geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch

from analysis.expanded.methods import build_mask
from analysis.expanded.paired_analysis import paired
from src.data import load_windows
from src.evaluate import prediction_metrics
from src.model_hooks import projection, outgoing_norms

METHODS = ('L', 'W', 'W_centered')
REGIMES = {'prefix': (0, 16), 'suffix': (16, 128)}
GEOMETRY_REGIONS = {'prefix': (0, 16), 'scored': (16, 127)}
CALIBRATION = ROOT / 'results/depth_extension/calibration/gpt2_0/calibration.pt'
OUTPUT = ROOT / 'results/followup/localization'


def restricted_mask(mask, start, stop):
    if mask.ndim != 3 or not 0 <= start < stop <= mask.shape[1]:
        raise ValueError('Invalid token mask or positional regime')
    positions = torch.arange(mask.shape[1], device=mask.device)
    return mask & ((positions >= start) & (positions < stop))[None, :, None]


def geometry(activations, mask, weight, norms, start, stop):
    """Per-window mean net vector length and sum of individual lengths."""
    if activations.shape != mask.shape or weight.shape[0] != activations.shape[-1]:
        raise ValueError('Incompatible activation, mask, or GPT-2 projection shapes')
    removed = activations[:, start:stop].float() * mask[:, start:stop]
    net = torch.linalg.vector_norm(removed @ weight.float(), dim=-1)
    individual = (removed.abs() * norms).sum(dim=-1)
    return net.mean(dim=-1), individual.mean(dim=-1)


def clustered_difference(frame, column, first, second, replicates=1000):
    a = frame[frame.method == first][['window_id', 'cluster_id', column]]
    b = frame[frame.method == second][['window_id', 'cluster_id', column]]
    if set(a.window_id) != set(b.window_id) or a.window_id.duplicated().any() or b.window_id.duplicated().any():
        raise ValueError('Paired window sets must match exactly')
    pair = a.merge(b, on=['window_id', 'cluster_id'], suffixes=('_a', '_b'), validate='one_to_one')
    if len(pair) != len(a):
        raise ValueError('Paired article identifiers must match')
    difference = pair[column + '_a'] - pair[column + '_b']
    groups = pd.DataFrame({'cluster_id': pair.cluster_id, 'delta': difference}).groupby('cluster_id').delta.agg(['sum', 'count'])
    rng = np.random.default_rng(1729)
    sample = rng.integers(0, len(groups), size=(replicates, len(groups)))
    draws = groups['sum'].to_numpy()[sample].sum(axis=1) / groups['count'].to_numpy()[sample].sum(axis=1)
    return {'first': first, 'second': second, 'measure': column, 'difference': difference.mean(),
            'low': np.quantile(draws, .025), 'high': np.quantile(draws, .975),
            'articles': len(groups), 'windows': len(pair), 'replicates': replicates}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch size must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Output already exists; choose a new directory')
    calibration = torch.load(CALIBRATION, map_location='cpu', weights_only=False)
    metadata = calibration['metadata']
    if metadata['split'] != 'train' or metadata['layer'] != 0:
        raise ValueError('Calibration must be GPT-2 layer-0 train data')
    tokenizer = AutoTokenizer.from_pretrained(metadata['model'], revision=metadata['model_revision'])
    model = AutoModelForCausalLM.from_pretrained(metadata['model'], revision=metadata['model_revision'], torch_dtype=torch.float32).to(args.device).eval()
    if model.config._commit_hash != metadata['model_commit'] or tokenizer.init_kwargs.get('_commit_hash') != metadata['tokenizer_commit']:
        raise ValueError('Model or tokenizer does not match training calibration')
    windows = load_windows(tokenizer, 'test', 128, 128, metadata['dataset_revision'])
    existing = pd.read_parquet(ROOT / 'results/depth_extension/test/gpt2_0/metrics.parquet')
    if set(w.window_id for w in windows) != set(existing.window_id):
        raise ValueError('Held-out windows differ from original depth analysis')
    module = projection(model, 'gpt2', 0)
    norms = outgoing_norms(model, 'gpt2', 0).to(args.device)
    weight = module.weight.detach().to(args.device).float()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = ROOT / 'analysis/followup/localization_protocol.md'
    manifest = {'scope': 'exploratory, selected after original held-out findings',
                'model': metadata['model'], 'model_commit': metadata['model_commit'],
                'dataset_revision': metadata['dataset_revision'], 'calibration_sha256': digest(CALIBRATION),
                'protocol_sha256': digest(protocol), 'code_sha256': digest(Path(__file__)),
                'split': 'test', 'layer': 0, 'fraction': .75, 'methods': list(METHODS),
                'regimes': REGIMES, 'geometry_regions': GEOMETRY_REGIONS,
                'score_positions': [16, 126], 'window_ids': [w.window_id for w in windows]}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    metrics_rows, geometry_rows = [], []
    for offset in range(0, len(windows), args.batch_size):
        batch = windows[offset:offset + args.batch_size]
        ids = torch.tensor([w.input_ids for w in batch], dtype=torch.long, device=args.device)
        captured = []
        def capture(_module, inputs):
            captured.append(inputs[0].detach())
        with torch.inference_mode():
            handle = module.register_forward_pre_hook(capture)
            try:
                baseline = model(ids, use_cache=False).logits
            finally:
                handle.remove()
            if len(captured) != 1:
                raise RuntimeError('Expected exactly one baseline projection input')
            activations = captured[0]
            for method in METHODS:
                mask = build_mask(activations, method, .75, calibration['thresholds'][.75], norms, calibration['means'])
                if not torch.all(mask.sum(-1) == int(.75 * activations.shape[-1])):
                    raise RuntimeError('Per-token removal quota mismatch')
                for region, (start, stop) in GEOMETRY_REGIONS.items():
                    net, individual = geometry(activations, mask, weight, norms, start, stop)
                    for i, window in enumerate(batch):
                        geometry_rows.append({'window_id': window.window_id, 'cluster_id': window.passage_id,
                                              'method': method, 'region': region,
                                              'net_removed_norm': float(net[i].cpu()),
                                              'sum_individual_norms': float(individual[i].cpu())})
                for regime, (start, stop) in REGIMES.items():
                    def intervention(_module, inputs):
                        actual = inputs[0]
                        current = build_mask(actual, method, .75, calibration['thresholds'][.75], norms, calibration['means'])
                        limited = restricted_mask(current, start, stop)
                        return (torch.where(limited, torch.zeros_like(actual), actual), *inputs[1:])
                    handle = module.register_forward_pre_hook(intervention)
                    try:
                        changed = model(ids, use_cache=False).logits
                    finally:
                        handle.remove()
                    outcome = prediction_metrics(baseline, changed, ids)
                    for i, window in enumerate(batch):
                        metrics_rows.append({'window_id': window.window_id, 'cluster_id': window.passage_id,
                                             'method': method, 'regime': regime,
                                             'removed_fraction_all_positions': .75 * (stop - start) / 128,
                                             **{k: float(v[i].cpu()) for k, v in outcome.items()}})
        print(f'{min(offset + args.batch_size, len(windows))}/128 windows', flush=True)
    metrics = pd.DataFrame(metrics_rows)
    geo = pd.DataFrame(geometry_rows)
    if not np.isfinite(metrics.select_dtypes('number')).all().all() or not np.isfinite(geo.select_dtypes('number')).all().all():
        raise ValueError('Non-finite metric')
    metrics.to_parquet(args.output / 'metrics.parquet', index=False)
    geo.to_parquet(args.output / 'geometry.parquet', index=False)
    all_positions = existing[existing.method.isin(METHODS)].copy()
    all_positions['regime'] = 'all'
    all_positions['removed_fraction_all_positions'] = all_positions.observed_fraction
    combined = pd.concat([metrics, all_positions[metrics.columns]], ignore_index=True)
    contrasts = []
    for regime, group in combined.groupby('regime'):
        for a, b in [('W', 'L'), ('W_centered', 'W')]:
            contrasts.append({'kind': 'intervention', 'regime': regime, **clustered_difference(group, 'kl', a, b)})
    for region, group in geo.groupby('region'):
        for measure in ('net_removed_norm', 'sum_individual_norms'):
            for a, b in [('W', 'L'), ('W_centered', 'W')]:
                contrasts.append({'kind': 'local_geometry', 'regime': region, **clustered_difference(group, measure, a, b)})
    pd.DataFrame(contrasts).to_csv(args.output / 'paired_contrasts.csv', index=False)
    print(f'Saved results to {args.output}', flush=True)


if __name__ == '__main__':
    main()
