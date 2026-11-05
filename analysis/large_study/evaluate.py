#!/usr/bin/env python3
"""Train-calibrated, resumable layer ablations on article-sampled windows."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from analysis.expanded.methods import METHODS, build_mask, fit_calibration
from scripts.ablate import MODELS, file_digest, validate_hook
from src.data import load_windows
from src.evaluate import prediction_metrics
from src.masks import condition_seed, random_mask
from src.model_hooks import collect_activations, outgoing_norms, projection

LAYERS = {'gpt2': (0, 2, 6, 8, 11), 'smollm2': (0, 5, 16, 21, 29)}
FRACTIONS = (.1, .2, .3, .4, .5, .6, .7, .75, .8, .85, .9, .95, .99, .995, .999)
METRICS = ('kl', 'baseline_nll', 'nll_delta', 'flip_rate', 'observed_fraction',
           'removed_abs_fraction', 'removed_squared_fraction', 'removed_weighted_fraction',
           'removed_centered_abs_fraction', 'removed_centered_weighted_fraction')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare_scores(activations, norms, means):
    """Cache magnitudes and stable local rankings from the unmodified forward pass."""
    absolute = activations.float().abs()
    centered = (activations.float() - means.to(activations.device)).abs()
    scores = {'L': absolute, 'W': absolute * norms, 'L_centered': centered,
              'W_centered': centered * norms}
    ranks = {}
    for name, values in scores.items():
        order = torch.argsort(values.cpu(), dim=-1, stable=True)
        rank = torch.empty_like(order)
        indices = torch.arange(order.shape[-1]).expand_as(order)
        rank.scatter_(-1, order, indices)
        ranks[name] = rank.to(activations.device)
    return absolute, centered, ranks


def make_mask(method, fraction, absolute, centered, ranks, calibration, windows, model_name, layer):
    channels = absolute.shape[-1]
    if method == 'identity':
        return torch.zeros_like(absolute, dtype=torch.bool)
    if method == 'all_zero':
        return torch.ones_like(absolute, dtype=torch.bool)
    if method in ranks:
        return ranks[method] < math.floor(fraction * channels)
    values = centered if method.endswith('_centered') else absolute
    threshold = torch.as_tensor(calibration['thresholds'][fraction][method], device=values.device, dtype=values.dtype)
    return values < threshold


def mask_statistics(mask, absolute, centered, norms):
    axes = tuple(range(1, absolute.ndim))
    result = {'observed_fraction': mask.float().mean(dim=axes)}
    for name, values in [('removed_abs_fraction', absolute), ('removed_squared_fraction', absolute.square()),
                         ('removed_weighted_fraction', absolute * norms), ('removed_centered_abs_fraction', centered),
                         ('removed_centered_weighted_fraction', centered * norms)]:
        total = values.sum(dim=axes)
        result[name] = torch.where(total > 0, (values * mask).sum(dim=axes) / total, torch.zeros_like(total))
    return result


def summarize(rows, replicates=1000, seed=0):
    """Resample articles, retaining all windows and weighting them equally."""
    frame = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    records = []
    for (model, layer, method, fraction), group in frame.groupby(['model', 'layer', 'method', 'fraction'], sort=True):
        cluster = group.groupby('cluster_id')[list(METRICS)].agg(['sum', 'count'])
        indices = rng.integers(0, len(cluster), size=(replicates, len(cluster)))
        for metric in METRICS:
            sums = cluster[(metric, 'sum')].to_numpy()
            counts = cluster[(metric, 'count')].to_numpy()
            estimates = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
            records.append(dict(model=model, layer=layer, method=method, fraction=fraction, metric=metric,
                                mean=float(group[metric].mean()), lower=float(np.quantile(estimates, .025)),
                                upper=float(np.quantile(estimates, .975)), clusters=len(cluster), replicates=replicates))
    return pd.DataFrame(records)



def paired_comparisons(rows, replicates=1000, seed=1):
    frame = pd.DataFrame(rows)
    pairs = [(method, 'random') for method in METHODS]
    pairs += [('W', 'L'), ('N', 'G'), ('W_centered', 'L_centered'), ('N_centered', 'G_centered')]
    pairs += [(method + '_centered', method) for method in ('N', 'L', 'G', 'W')]
    rng = np.random.default_rng(seed)
    records = []
    for (model, layer, fraction), group in frame.groupby(['model', 'layer', 'fraction']):
        for method, reference in pairs:
            left = group[group.method == method]
            right = group[group.method == reference]
            if left.empty or right.empty:
                continue
            paired = left.merge(right, on=['window_id', 'cluster_id'], suffixes=('_left', '_right'), validate='one_to_one')
            if len(paired) != len(left) or len(paired) != len(right):
                raise ValueError('paired windows differ')
            for metric in ('kl', 'observed_fraction'):
                paired['difference'] = paired[metric + '_left'] - paired[metric + '_right']
                cluster = paired.groupby('cluster_id').difference.agg(['sum', 'count'])
                indices = rng.integers(0, len(cluster), size=(replicates, len(cluster)))
                estimates = cluster['sum'].to_numpy()[indices].sum(axis=1) / cluster['count'].to_numpy()[indices].sum(axis=1)
                records.append(dict(model=model, layer=layer, method=method, reference=reference, fraction=fraction,
                                    metric=metric, mean=float(paired.difference.mean()), lower=float(np.quantile(estimates, .025)),
                                    upper=float(np.quantile(estimates, .975)), clusters=len(cluster), replicates=replicates))
    return pd.DataFrame(records)


def calibrate(model_name, layer, model, tokenizer, args):
    destination = args.output / 'calibration' / f'{model_name}_{layer}.pt'
    source = args.source / 'descriptive' / f'{model_name}_{layer}'
    if not (source / 'activations.npy').exists():
        depth = ROOT / 'results/depth_extension/calibration' / f'{model_name}_{layer}'
        if (depth / 'activations.npy').exists():
            source = depth
    if not (source / 'activations.npy').exists():
        source = args.output / 'training' / f'{model_name}_{layer}'
        source.mkdir(parents=True, exist_ok=True)
        if not (source / 'metadata.json').exists():
            windows = load_windows(tokenizer, 'train', 128, 128, args.dataset_revision)
            collect_activations(model, model_name, layer, windows, source / 'activations.npy', args.batch_size, args.device)
            write_json(source / 'metadata.json', {'split': 'train', 'model': MODELS[model_name], 'layer': layer,
                       'model_revision': args.model_revision, 'dataset_revision': args.dataset_revision,
                       'model_commit': getattr(model.config, '_commit_hash', None),
                       'tokenizer_commit': tokenizer.init_kwargs.get('_commit_hash'),
                       'windows': [w.window_id for w in windows]})
    metadata_path = source / 'metadata.json'
    metadata = (json.loads(metadata_path.read_text()) if metadata_path.exists() else
                torch.load(source / 'calibration.pt', map_location='cpu', weights_only=False)['metadata'])
    if metadata['split'] != 'train' or metadata['model'] != MODELS[model_name] or metadata['layer'] != layer or metadata['dataset_revision'] != args.dataset_revision:
        raise ValueError('invalid training provenance')
    if metadata['model_commit'] != getattr(model.config, '_commit_hash', None) or metadata['tokenizer_commit'] != tokenizer.init_kwargs.get('_commit_hash'):
        raise ValueError('model or tokenizer differs from training activations')
    digest = file_digest(source / 'activations.npy')
    fitting_hashes = {str(p.relative_to(ROOT)): file_digest(p) for p in (ROOT / 'analysis/expanded/methods.py', ROOT / 'src/calibrate.py')}
    if destination.exists():
        calibration = torch.load(destination, map_location='cpu', weights_only=False)
        if calibration['activation_sha256'] != digest or calibration['metadata'] != metadata or calibration.get('fitting_hashes') != fitting_hashes or tuple(calibration['thresholds']) != (0., *FRACTIONS, 1.):
            raise ValueError('calibration provenance differs')
        return calibration, destination
    calibration = fit_calibration(np.load(source / 'activations.npy', mmap_mode='r'), fractions=(0., *FRACTIONS, 1.))
    calibration.update(metadata=metadata, activation_sha256=digest, fitting_hashes=fitting_hashes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(calibration, destination)
    return calibration, destination


def run_layer(model_name, layer, model, windows, calibration, calibration_path, args, manifest):
    directory = args.output / args.stage / f'{model_name}_{layer}'
    directory.mkdir(parents=True, exist_ok=True)
    module = projection(model, model_name, layer)
    norms = outgoing_norms(model, model_name, layer).to(args.device)
    conditions = [('identity', 0.)] + [(m, p) for p in FRACTIONS for m in (*METHODS, 'random')] + [('all_zero', 1.)]
    output_conditions = conditions + [(m, 1.) for m in (*METHODS, 'random')]
    signature = {'model': model_name, 'layer': layer, 'stage': args.stage, 'batch_size': args.batch_size,
                 'device': args.device, 'calibration_sha256': file_digest(calibration_path),
                 'windows': [{'id': w.window_id, 'article': w.passage_id, 'tokens': w.input_ids} for w in windows],
                 'manifest': manifest, 'conditions': conditions,
                 'source_sha256': {str(p.relative_to(ROOT)): file_digest(p) for p in
                     [Path(__file__), ROOT / 'analysis/large_study/data.py', ROOT / 'analysis/expanded/methods.py',
                      ROOT / 'src/masks.py', ROOT / 'src/evaluate.py', ROOT / 'src/calibrate.py', ROOT / 'src/model_hooks.py']}}
    signature = json.loads(json.dumps(signature))
    path = directory / 'signature.json'
    if path.exists() and json.loads(path.read_text()) != signature:
        raise ValueError('checkpoint signature changed; choose a new output directory')
    if not path.exists():
        write_json(path, signature)
    validate_hook(model, module, torch.tensor([windows[0].input_ids], device=args.device), len(norms))
    rows = []
    for offset in range(0, len(windows), args.batch_size):
        batch = windows[offset:offset + args.batch_size]
        checkpoint = directory / f'batch_{offset:05d}.json'
        if checkpoint.exists():
            batch_rows = json.loads(checkpoint.read_text())
            expected = {(w.window_id, m, p) for w in batch for m, p in output_conditions}
            if len(batch_rows) != len(expected) or {(r['window_id'], r['method'], r['fraction']) for r in batch_rows} != expected or any(not np.isfinite(r[k]) for r in batch_rows for k in METRICS):
                raise ValueError('invalid checkpoint')
            rows.extend(batch_rows)
            continue
        batch_started = time.monotonic()
        ids = torch.tensor([w.input_ids for w in batch], device=args.device)
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
                raise RuntimeError('projection capture failed')
            absolute, centered, ranks = prepare_scores(captured[0], norms, calibration['means'])
            random_ranks = []
            for window in batch:
                generator = torch.Generator(device='cpu').manual_seed(
                    condition_seed(model_name, layer, window.window_id, 'random', 0., 0))
                orders = torch.stack([torch.randperm(absolute.shape[-1], generator=generator)
                                      for _ in range(absolute.shape[1])])
                rank = torch.empty_like(orders)
                rank.scatter_(-1, orders, torch.arange(orders.shape[-1]).expand_as(orders))
                random_ranks.append(rank)
            ranks['random'] = torch.stack(random_ranks).to(absolute.device)
            batch_rows = []
            for method, fraction in conditions:
                mask = make_mask(method, fraction, absolute, centered, ranks, calibration, batch, model_name, layer)
                if args.stage == 'validation':
                    if method in METHODS:
                        reference = build_mask(captured[0], method, fraction, calibration['thresholds'][fraction], norms, calibration['means'])
                        if not torch.equal(mask, reference):
                            raise ValueError('cached mask differs from direct mask')
                    if method == 'random' or method.startswith(('L', 'W')):
                        if not torch.all(mask.sum(-1) == math.floor(fraction * absolute.shape[-1])):
                            raise ValueError('fixed-budget mask has incorrect count')
                statistics = mask_statistics(mask, absolute, centered, norms)
                if method == 'identity' and args.stage != 'validation':
                    changed = baseline
                else:
                    def intervene(_module, inputs):
                        return (torch.where(mask, torch.zeros_like(inputs[0]), inputs[0]), *inputs[1:])
                    handle = module.register_forward_pre_hook(intervene)
                    try:
                        changed = model(ids, use_cache=False).logits
                    finally:
                        handle.remove()
                metrics = {**prediction_metrics(baseline, changed, ids), **statistics}
                if args.stage == 'validation' and method == 'identity' and metrics['kl'].abs().max() > 1e-6:
                    raise ValueError('identity changed predictions')
                if args.stage == 'validation' and method == 'all_zero' and not mask.all():
                    raise ValueError('all-zero mask is incomplete')
                for i, window in enumerate(batch):
                    values = {k: float(v[i].cpu()) for k, v in metrics.items()}
                    if not all(np.isfinite(v) for v in values.values()):
                        raise ValueError('non-finite metric')
                    batch_rows.append(dict(model=model_name, layer=layer, window_id=window.window_id,
                                           cluster_id=window.passage_id, method=method, fraction=fraction, **values))
        endpoints = [row for row in batch_rows if row['method'] == 'all_zero']
        batch_rows.extend({**row, 'method': method} for row in endpoints for method in (*METHODS, 'random'))
        write_json(checkpoint, batch_rows)
        rows.extend(batch_rows)
        print(f'{model_name} layer {layer}: {min(offset + args.batch_size, len(windows))}/{len(windows)}', flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(directory / 'metrics.csv', index=False)
    table.to_parquet(directory / 'metrics.parquet', index=False)
    summarize(rows).to_csv(directory / 'summary.csv', index=False)
    paired_comparisons(rows).to_csv(directory / 'paired_comparisons.csv', index=False)
    write_json(directory / 'run.json', {'status': 'passed', 'window_count': len(windows),
               'article_count': len({w.passage_id for w in windows}), 'calibration_sha256': file_digest(calibration_path),
               'signature_sha256': file_digest(path), 'model_commit': calibration['metadata']['model_commit'],
               'tokenizer_commit': calibration['metadata']['tokenizer_commit'], 'sample_manifest': manifest, 'random_control': 'nested prefixes of independently seeded per-token permutations; floor(fraction * channels)',
               'threshold_rule': 'strict less than; N and G realized fractions can differ from random'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['validation', 'test'])
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--layers', type=int, nargs='+')
    parser.add_argument('--count', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--source', type=Path, default=ROOT / 'results/study')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/large_study')
    parser.add_argument('--model-revision', default='main')
    parser.add_argument('--dataset-revision', default='b08601e04326c79dfdd32d625aee71d232d685c3')
    args = parser.parse_args()
    if args.count < 1 or args.batch_size < 1 or any(layer not in LAYERS[args.model] for layer in (args.layers or ())):
        parser.error('invalid count, batch size, or layer')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from analysis.large_study.data import load_sample
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model], revision=args.model_revision)
    model = AutoModelForCausalLM.from_pretrained(MODELS[args.model], revision=args.model_revision, torch_dtype=torch.float32).to(args.device).eval()
    if args.stage == 'test':
        for name, layer in [('gpt2', 6), ('smollm2', 16)]:
            validation = args.output / 'validation' / f'{name}_{layer}'
            record = json.loads((validation / 'run.json').read_text())
            signature = json.loads((validation / 'signature.json').read_text())
            if (record['status'] != 'passed' or record['window_count'] < 4 or
                record['signature_sha256'] != file_digest(validation / 'signature.json') or
                any(file_digest(ROOT / p) != h for p, h in signature['source_sha256'].items()) or
                signature['calibration_sha256'] != file_digest(args.output / 'calibration' / f'{name}_{layer}.pt') or
                signature['manifest']['dataset_revision'] != args.dataset_revision):
                raise ValueError('both models require current real validation before test')
    if args.stage == 'validation':
        windows = load_windows(tokenizer, 'validation', args.count, 128, args.dataset_revision)
        manifest = {'split': 'validation', 'dataset_revision': args.dataset_revision}
    else:
        tokenizers = {name: (tokenizer if name == args.model else AutoTokenizer.from_pretrained(model_id, revision=args.model_revision))
                      for name, model_id in MODELS.items()}
        windows_by_model, manifest = load_sample(tokenizers, count=args.count, revision=args.dataset_revision)
        manifest_path = args.output / 'sample_manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('sample manifest differs; use a new output directory')
        if not manifest_path.exists():
            write_json(manifest_path, manifest)
        windows = windows_by_model[args.model]
    for layer in args.layers or LAYERS[args.model]:
        calibration, path = calibrate(args.model, layer, model, tokenizer, args)
        run_layer(args.model, layer, model, windows, calibration, path, args, manifest)


if __name__ == '__main__':
    main()
