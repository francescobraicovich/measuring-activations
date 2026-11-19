#!/usr/bin/env python3
"""Evaluate sequential ablations at every MLP layer on held-out windows."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from analysis.expanded.methods import METHODS, build_mask, fit_calibration
from analysis.large_study.data import DATASET_REVISION, load_sample
from analysis.large_study.evaluate import FRACTIONS, METRICS, mask_statistics, paired_comparisons, summarize, write_json
from scripts.ablate import MODELS, file_digest
from src.data import load_windows
from src.evaluate import prediction_metrics
from src.masks import condition_seed
from src.model_hooks import collect_activations, outgoing_norms, projection


def sequential_forward(model, modules, calibrations, norms, ids, windows, model_name, method, fraction):
    """Select from the current activation at each layer during one forward pass."""
    statistics = []
    handles = []
    for layer, module in enumerate(modules):
        def intervene(_module, inputs, layer=layer):
            activation = inputs[0]
            calibration = calibrations[layer]
            if method == 'identity':
                mask = torch.zeros_like(activation, dtype=torch.bool)
            elif method == 'all_zero':
                mask = torch.ones_like(activation, dtype=torch.bool)
            elif method == 'random':
                count = math.floor(fraction * activation.shape[-1])
                masks = []
                for window in windows:
                    generator = torch.Generator(device='cpu').manual_seed(
                        condition_seed(model_name, layer, window.window_id, 'random', 0., 0))
                    order = torch.stack([torch.randperm(activation.shape[-1], generator=generator)
                                         for _ in range(activation.shape[1])])
                    selected = torch.zeros_like(order, dtype=torch.bool)
                    selected.scatter_(-1, order[:, :count], True)
                    masks.append(selected)
                mask = torch.stack(masks).to(activation.device)
            else:
                mask = build_mask(activation, method, fraction, calibration['thresholds'][fraction],
                                  norms[layer], calibration['means'])
            absolute = activation.float().abs()
            centered = (activation.float() - calibration['means'].to(activation.device)).abs()
            statistics.append(mask_statistics(mask, absolute, centered, norms[layer]))
            return (activation.masked_fill(mask, 0), *inputs[1:])
        handles.append(module.register_forward_pre_hook(intervene))
    try:
        logits = model(ids, use_cache=False).logits
    finally:
        for handle in handles:
            handle.remove()
    if len(statistics) != len(modules):
        raise RuntimeError('Every layer must be visited exactly once')
    return logits, {key: torch.stack([item[key] for item in statistics]).mean(0) for key in statistics[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--count', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--fractions', type=float, nargs='+', default=[.1, .5, .75, .9, .99])
    parser.add_argument('--output', type=Path, default=ROOT / 'results/large_study/all_layers')
    args = parser.parse_args()
    if args.count < 1 or args.batch_size < 1 or len(set(args.fractions)) != len(args.fractions) or any(p not in FRACTIONS for p in args.fractions):
        parser.error('count and batch size must be positive; fractions must be distinct study fractions')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizers = {name: AutoTokenizer.from_pretrained(model_id) for name, model_id in MODELS.items()}
    model = AutoModelForCausalLM.from_pretrained(MODELS[args.model], torch_dtype=torch.float32).to(args.device).eval()
    windows_by_model, manifest = load_sample(tokenizers, count=args.count)
    windows = windows_by_model[args.model]
    original_manifest = json.loads((ROOT / 'results/large_study/sample_manifest.json').read_text())
    if (manifest['dataset_text_sha256'] != original_manifest['dataset_text_sha256'] or
            manifest['models'][args.model]['windows'] != original_manifest['models'][args.model]['windows'][:args.count]):
        raise ValueError('Held-out sample differs from the original study')
    layer_count = model.config.n_layer if args.model == 'gpt2' else model.config.num_hidden_layers
    modules = [projection(model, args.model, layer) for layer in range(layer_count)]
    norms = [outgoing_norms(model, args.model, layer).to(args.device) for layer in range(layer_count)]
    destination = args.output / args.model
    destination.mkdir(parents=True, exist_ok=True)
    calibrations, calibration_hashes = [], []
    training = None
    for layer in range(layer_count):
        original = ROOT / 'results/large_study/calibration' / f'{args.model}_{layer}.pt'
        path = original if original.exists() else destination / f'calibration_{layer}.pt'
        if not path.exists():
            if training is None:
                training = load_windows(tokenizers[args.model], 'train', 128, 128, DATASET_REVISION)
            activation_path = destination / 'training_activations.npy'
            collect_activations(model, args.model, layer, training, activation_path, args.batch_size, args.device)
            values = np.load(activation_path, mmap_mode='r')
            calibration = fit_calibration(values, fractions=(0., *sorted(args.fractions), 1.))
            calibration['metadata'] = {'split': 'train', 'model': MODELS[args.model], 'layer': layer,
                                       'dataset_revision': DATASET_REVISION,
                                       'model_commit': getattr(model.config, '_commit_hash', None),
                                       'tokenizer_commit': tokenizers[args.model].init_kwargs.get('_commit_hash'),
                                       'windows': [w.window_id for w in training]}
            calibration['activation_sha256'] = file_digest(activation_path)
            del values
            activation_path.unlink()
            torch.save(calibration, path)
            print(f'{args.model}: calibrated layer {layer}', flush=True)
        calibration = torch.load(path, map_location='cpu', weights_only=False)
        metadata = calibration['metadata']
        if (metadata['split'] != 'train' or metadata['model'] != MODELS[args.model] or metadata['layer'] != layer
                or metadata['dataset_revision'] != DATASET_REVISION
                or metadata['model_commit'] != getattr(model.config, '_commit_hash', None)
                or metadata['tokenizer_commit'] != tokenizers[args.model].init_kwargs.get('_commit_hash')):
            raise ValueError(f'Calibration provenance differs at layer {layer}')
        if not set(args.fractions).issubset(calibration['thresholds']):
            raise ValueError(f'Calibration is missing requested fractions at layer {layer}')
        calibrations.append(calibration)
        calibration_hashes.append(file_digest(path))
    signature = {'model': args.model, 'model_commit': getattr(model.config, '_commit_hash', None),
                 'device': args.device, 'batch_size': args.batch_size, 'fractions': args.fractions,
                 'manifest': manifest, 'calibration_sha256': calibration_hashes,
                 'source_sha256': {str(path.relative_to(ROOT)): file_digest(path) for path in
                                   (Path(__file__), ROOT / 'analysis/expanded/methods.py', ROOT / 'src/masks.py',
                                    ROOT / 'src/evaluate.py', ROOT / 'src/model_hooks.py')}}
    signature_path = destination / 'signature.json'
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError('Run configuration changed; choose a new output directory')
    if not signature_path.exists():
        write_json(signature_path, signature)
    conditions = [('identity', 0.)] + [(method, fraction) for fraction in args.fractions for method in (*METHODS, 'random')] + [('all_zero', 1.)]
    rows = []
    for offset in range(0, len(windows), args.batch_size):
        batch = windows[offset:offset + args.batch_size]
        ids = torch.tensor([w.input_ids for w in batch], device=args.device)
        checkpoint = destination / f'batch_{offset:05d}.json'
        if checkpoint.exists():
            batch_rows = json.loads(checkpoint.read_text())
            expected = {(w.window_id, method, p) for w in batch for method, p in conditions}
            if len(batch_rows) != len(expected) or {(r['window_id'], r['method'], r['fraction']) for r in batch_rows} != expected or any(not np.isfinite(r[k]) for r in batch_rows for k in METRICS):
                raise ValueError('Invalid checkpoint')
            rows.extend(batch_rows)
            continue
        batch_rows = []
        with torch.inference_mode():
            baseline = model(ids, use_cache=False).logits
            for method, fraction in conditions:
                changed, statistics = sequential_forward(model, modules, calibrations, norms, ids, batch, args.model, method, fraction)
                metrics = {**prediction_metrics(baseline, changed, ids), **statistics}
                if method == 'identity' and metrics['kl'].abs().max() > 1e-6:
                    raise ValueError('Identity changed predictions')
                for i, window in enumerate(batch):
                    values = {key: float(value[i].cpu()) for key, value in metrics.items()}
                    if not all(np.isfinite(value) for value in values.values()):
                        raise ValueError('Non-finite measurement')
                    batch_rows.append(dict(model=args.model, layer=-1, window_id=window.window_id,
                                           cluster_id=window.passage_id, method=method, fraction=fraction, **values))
        write_json(checkpoint, batch_rows)
        rows.extend(batch_rows)
        print(f'{args.model}: evaluated {offset + len(batch)}/{len(windows)} windows', flush=True)
    endpoints = [row for row in rows if row['method'] == 'all_zero']
    rows.extend({**row, 'method': method} for row in endpoints for method in (*METHODS, 'random'))
    pd.DataFrame(rows).to_parquet(destination / 'metrics.parquet', index=False)
    summarize(rows).to_csv(destination / 'summary.csv', index=False)
    paired_comparisons(rows).to_csv(destination / 'paired_comparisons.csv', index=False)
    write_json(destination / 'run.json', {'status': 'passed', 'windows': len(windows),
               'articles': len({w.passage_id for w in windows}), 'layers': layer_count,
               'signature_sha256': file_digest(signature_path)})


if __name__ == '__main__':
    main()
