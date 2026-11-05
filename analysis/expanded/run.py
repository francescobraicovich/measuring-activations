#!/usr/bin/env python3
"""Run exploratory centered and uncentered ablations with train-only calibration."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from analysis.expanded.methods import FRACTIONS, build_mask, condition_seed, fit_calibration, random_mask
from scripts.ablate import LAYERS, MODELS, file_digest, paired_summary, validate_hook
from src.data import load_windows
from src.evaluate import prediction_metrics
from src.model_hooks import outgoing_norms, projection

METHODS = ('N', 'L', 'G', 'W', 'N_centered', 'L_centered', 'G_centered', 'W_centered')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def calibrate(source, output):
    output.mkdir(parents=True, exist_ok=True)
    gate_path = output / 'train_complete.json'
    if gate_path.exists():
        existing = json.loads(gate_path.read_text())
        for entry in existing['entries']:
            if file_digest(entry['path']) != entry['sha256']:
                raise ValueError('Completed calibration changed')
        print('Reusing completed train calibration', flush=True)
        return
    entries = []
    for model, layers in LAYERS.items():
        for layer in layers:
            directory = source / 'descriptive' / f'{model}_{layer}'
            metadata = json.loads((directory / 'metadata.json').read_text())
            if metadata['split'] != 'train' or metadata['model'] != MODELS[model] or metadata['layer'] != layer:
                raise ValueError('Invalid train provenance')
            path = output / f'{model}_{layer}.pt'
            activation_path = directory / 'activations.npy'
            calibration = fit_calibration(np.load(activation_path, mmap_mode='r'), fractions=FRACTIONS, split='train')
            calibration['metadata'] = metadata
            calibration['activation_sha256'] = file_digest(activation_path)
            torch.save(calibration, path)
            entries.append({'model': model, 'layer': layer, 'path': str(path.resolve()), 'sha256': file_digest(path)})
    write_json(output / 'train_complete.json', {'entries': entries, 'scope': 'exploratory', 'calibration_code': {str(p.relative_to(ROOT)): file_digest(p) for p in (ROOT / 'analysis/expanded/methods.py', ROOT / 'src/calibrate.py')}})


def evaluate(model, module, ids, windows, name, layer, method, fraction, calibration, norms, random_control, repetition, baseline):
    statistics = {}
    def hook(_module, inputs):
        activations = inputs[0]
        mask = build_mask(activations, method, fraction, calibration['thresholds'].get(fraction), norms, calibration['means'])
        if random_control:
            counts = mask.sum(-1)
            mask = torch.stack([random_mask(counts[i], activations.shape[-1],
                condition_seed(name, layer, window.window_id, method, fraction, repetition))
                for i, window in enumerate(windows)])
            if not torch.equal(mask.sum(-1), counts):
                raise RuntimeError('Random control count mismatch')
        absolute = activations.float().abs()
        centered = (activations.float() - calibration['means'].to(activations.device)).abs()
        axes = tuple(range(1, activations.ndim))
        statistics['observed_fraction'] = mask.float().mean(dim=axes)
        for label, values in [('removed_abs_fraction', absolute), ('removed_squared_fraction', absolute.square()),
                              ('removed_weighted_fraction', absolute * norms),
                              ('removed_centered_abs_fraction', centered),
                              ('removed_centered_weighted_fraction', centered * norms)]:
            total = values.sum(dim=axes)
            statistics[label] = torch.where(total > 0, (values * mask).sum(dim=axes) / total, torch.zeros_like(total))
        return (torch.where(mask, torch.zeros_like(activations), activations), *inputs[1:])
    with torch.inference_mode():
        handle = module.register_forward_pre_hook(hook)
        try:
            changed = model(ids, use_cache=False).logits
        finally:
            handle.remove()
        metrics = prediction_metrics(baseline, changed, ids)
    rows = [{**{k: float(v[i].cpu()) for k, v in metrics.items()},
             **{k: float(v[i].cpu()) for k, v in statistics.items()}} for i in range(len(windows))]
    if not all(np.isfinite(value) for row in rows for value in row.values()):
        raise ValueError('Non-finite evaluation metric')
    return rows


def run_one(model_name, layer, split, count, batch_size, device, output, calibration_path, train_gate):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    calibration = torch.load(calibration_path, map_location='cpu', weights_only=False)
    metadata = calibration['metadata']
    model_id = MODELS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=metadata['model_revision'])
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=metadata['model_revision'], torch_dtype=torch.float32).to(device).eval()
    if tokenizer.init_kwargs.get('_commit_hash') != metadata['tokenizer_commit'] or getattr(model.config, '_commit_hash', None) != metadata['model_commit']:
        raise ValueError('Model or tokenizer commit differs from train calibration')
    windows = load_windows(tokenizer, split, count, 128, metadata['dataset_revision'])
    module = projection(model, model_name, layer)
    norms = outgoing_norms(model, model_name, layer).to(device)
    validate_hook(model, module, torch.tensor([windows[0].input_ids], device=device), len(norms))
    fractions = [p for p in FRACTIONS if 0 < p < 1] if LAYERS[model_name][layer] == 'middle' else [.75]
    conditions = [(m, p, False, 0) for p in fractions for m in METHODS]
    conditions += [(m, p, True, r) for p in fractions for m in METHODS for r in range(3 if p == .75 and LAYERS[model_name][layer] == 'middle' else 1)]
    conditions += [('L', 0., False, 0), ('L', 1., False, 0)]
    output.mkdir(parents=True, exist_ok=True)
    signature = {'model': model_name, 'layer': layer, 'split': split, 'count': count, 'batch_size': batch_size,
                 'device': device, 'calibration_sha256': file_digest(calibration_path), 'train_gate_sha256': file_digest(train_gate),
                 'windows': [w.window_id for w in windows], 'conditions': conditions,
                 'source_sha256': {str(p.relative_to(ROOT)): file_digest(p) for p in [Path(__file__), ROOT / 'analysis/expanded/methods.py', ROOT / 'src/masks.py', ROOT / 'src/evaluate.py', ROOT / 'src/data.py', ROOT / 'src/model_hooks.py', ROOT / 'src/calibrate.py', ROOT / 'scripts/ablate.py']}}
    signature = json.loads(json.dumps(signature))
    signature_path = output / 'signature.json'
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError('Checkpoint signature differs; use a new output directory')
    if not signature_path.exists():
        write_json(signature_path, signature)
    rows = []
    for offset in range(0, count, batch_size):
        batch = windows[offset:offset + batch_size]
        checkpoint = output / f'batch_{offset:04d}.json'
        if checkpoint.exists():
            batch_rows = json.loads(checkpoint.read_text())
            expected_keys = {(w.window_id, 'identity' if p == 0 else 'all_zero' if p == 1 else m + '_random' if random else m, p, r) for w in batch for m, p, random, r in conditions}
            actual_keys = {(row['window_id'], row['method'], row['fraction'], row['repetition']) for row in batch_rows}
            metric_names = ('kl', 'baseline_nll', 'nll_delta', 'flip_rate', 'observed_fraction', 'removed_abs_fraction', 'removed_squared_fraction', 'removed_weighted_fraction', 'removed_centered_abs_fraction', 'removed_centered_weighted_fraction')
            if len(batch_rows) != len(expected_keys) or actual_keys != expected_keys or any(not np.isfinite(row[key]) for row in batch_rows for key in metric_names):
                raise ValueError('Invalid batch checkpoint')
            rows.extend(batch_rows)
            continue
        ids = torch.tensor([w.input_ids for w in batch], device=device)
        with torch.inference_mode():
            baseline = model(ids, use_cache=False).logits
        batch_rows = []
        for method, fraction, random_control, repetition in conditions:
            values = evaluate(model, module, ids, batch, model_name, layer, method, fraction, calibration, norms, random_control, repetition, baseline)
            label = 'identity' if fraction == 0 else 'all_zero' if fraction == 1 else method + '_random' if random_control else method
            for window, metrics in zip(batch, values):
                if label == 'identity' and abs(metrics['kl']) > 1e-6:
                    raise RuntimeError('Identity intervention changed predictions')
                if label == 'all_zero' and metrics['observed_fraction'] != 1:
                    raise RuntimeError('All-zero intervention did not remove all activations')
                batch_rows.append(dict(model=model_name, layer=layer, depth=LAYERS[model_name][layer], window_id=window.window_id,
                                       cluster_id=window.passage_id, method=label, fraction=fraction, repetition=repetition, **metrics))
        write_json(checkpoint, batch_rows)
        rows.extend(batch_rows)
        print(f'{model_name} layer {layer} {split}: {min(offset + batch_size, count)}/{count} windows', flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / 'metrics.csv', index=False)
    table.to_parquet(output / 'metrics.parquet', index=False)
    paired_summary(rows).to_csv(output / 'paired_contrasts.csv', index=False)
    write_json(output / 'run.json', {'model': model_name, 'layer': layer, 'split': split, 'status': 'passed',
               'window_count': count, 'calibration_sha256': file_digest(calibration_path), 'scope': 'exploratory',
               'seed_scheme': 'SHA-256 condition tuple', 'calibration': metadata})



def compare_original(directory, original):
    expanded = pd.read_csv(directory / 'metrics.csv')
    baseline = pd.read_csv(original / 'metrics.csv')
    keys = ['model', 'layer', 'window_id', 'method', 'fraction', 'repetition']
    selected = expanded[expanded.method.isin(['N', 'L', 'G', 'W', 'N_random', 'L_random', 'G_random', 'W_random', 'identity', 'all_zero'])]
    paired = selected.merge(baseline, on=keys, suffixes=('_expanded', '_original'), validate='one_to_one')
    baseline = baseline[baseline.window_id.isin(selected.window_id.unique()) & ((baseline.fraction.isin([.25, .5, .75, .9])) | baseline.method.isin(['identity', 'all_zero']))]
    if len(paired) != len(baseline) or not len(paired):
        raise ValueError('Original comparison has missing or duplicate rows')
    columns = ['kl', 'baseline_nll', 'nll_delta', 'flip_rate', 'observed_fraction', 'removed_abs_fraction', 'removed_squared_fraction', 'removed_weighted_fraction']
    differences = {column: float((paired[column + '_expanded'] - paired[column + '_original']).abs().max()) for column in columns}
    write_json(directory / 'original_comparison.json', {'matched_rows': len(paired), 'maximum_absolute_differences': differences})
    if any(value > 1e-5 for value in differences.values()):
        raise ValueError('Original reproduction differs; inspect original_comparison.json')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['calibrate', 'validation', 'test'])
    parser.add_argument('--source', type=Path, default=ROOT / 'results/study')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/expanded')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--test-windows', type=int, choices=[64, 128], default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch size must be positive')
    args.output = args.output.resolve()
    train = args.output / 'calibration'
    if args.stage == 'calibrate':
        calibrate(args.source, train)
        return
    gate = json.loads((train / 'train_complete.json').read_text())
    expected = {(m, layer) for m, layers in LAYERS.items() for layer in layers}
    if {(entry['model'], entry['layer']) for entry in gate['entries']} != expected:
        raise ValueError('Train gate must cover all layers')
    for path, digest in gate['calibration_code'].items():
        if file_digest(ROOT / path) != digest:
            raise ValueError('Calibration code changed')
    for entry in gate['entries']:
        if file_digest(entry['path']) != entry['sha256']:
            raise ValueError('Train calibration changed')
        data = torch.load(entry['path'], map_location='cpu', weights_only=False)
        source_dir = args.source / 'descriptive' / f"{entry['model']}_{entry['layer']}"
        if data['activation_sha256'] != file_digest(source_dir / 'activations.npy') or data['metadata'] != json.loads((source_dir / 'metadata.json').read_text()):
            raise ValueError('Train source activations or metadata changed')
    if args.stage == 'test':
        description = json.loads((args.output / 'training_description.json').read_text())
        if description['split'] != 'train' or {(x['model'], x['layer']) for x in description['entries']} != expected:
            raise ValueError('Training descriptions must cover all layers')
        for entry in description['entries']:
            calibration_path = train / f"{entry['model']}_{entry['layer']}.pt"
            source_path = args.source / 'descriptive' / f"{entry['model']}_{entry['layer']}" / 'activations.npy'
            if entry['calibration_sha256'] != file_digest(calibration_path) or entry['activation_sha256'] != file_digest(source_path):
                raise ValueError('Training description provenance changed')
        validation = json.loads((args.output / 'validation_complete.json').read_text())
        if {(x['model'], x['layer']) for x in validation['entries']} != {('gpt2', 6), ('smollm2', 16)}:
            raise ValueError('Validation gate must cover both middle layers')
        for entry in validation['entries']:
            if file_digest(entry['path']) != entry['sha256']:
                raise ValueError('Validation record changed')
            record = json.loads(Path(entry['path']).read_text())
            expected_calibration = train / f"{entry['model']}_{entry['layer']}.pt"
            if record['status'] != 'passed' or record['split'] != 'validation' or record['window_count'] < 4 or record['calibration_sha256'] != file_digest(expected_calibration):
                raise ValueError('Validation record does not match train calibration')
    targets = [(m, layer) for m, layers in LAYERS.items() for layer in layers if args.stage == 'test' or LAYERS[m][layer] == 'middle']
    entries = []
    for model, layer in targets:
        directory = args.output / args.stage / f'{model}_{layer}'
        calibration_path = train / f'{model}_{layer}.pt'
        run_one(model, layer, args.stage, 4 if args.stage == 'validation' else args.test_windows, args.batch_size, args.device, directory, calibration_path, train / 'train_complete.json')
        if args.stage == 'test':
            compare_original(directory, args.source / 'test' / f'{model}_{layer}')
        record = directory / 'run.json'
        entries.append({'model': model, 'layer': layer, 'path': str(record), 'sha256': file_digest(record)})
    write_json(args.output / f'{args.stage}_complete.json', {'entries': entries, 'scope': 'exploratory'})


if __name__ == '__main__':
    main()
