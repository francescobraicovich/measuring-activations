#!/usr/bin/env python3
"""Run a separate exploratory depth extension using train-only calibration."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from analysis.expanded.methods import METHODS, build_mask, fit_calibration
from analysis.expanded.run import evaluate, write_json
from scripts.ablate import MODELS, file_digest, paired_summary, validate_hook
from src.data import load_windows
from src.model_hooks import collect_activations, outgoing_norms, projection

LAYERS = {'gpt2': (0, 4, 8, 10), 'smollm2': (0, 10, 21, 26)}
VALIDATION_LAYERS = {'gpt2': 4, 'smollm2': 21}
SOURCES = [Path(__file__), ROOT / 'analysis/expanded/methods.py', ROOT / 'analysis/expanded/run.py',
           ROOT / 'src/calibrate.py', ROOT / 'src/data.py', ROOT / 'src/model_hooks.py',
           ROOT / 'src/masks.py', ROOT / 'src/evaluate.py', ROOT / 'scripts/ablate.py']


def source_hashes():
    return {str(path.relative_to(ROOT)): file_digest(path) for path in SOURCES}


def model_and_windows(name, split, count, metadata, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = MODELS[name]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=metadata['model_revision'])
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=metadata['model_revision'], torch_dtype=torch.float32).to(device).eval()
    if getattr(model.config, '_commit_hash', None) != metadata['model_commit'] or tokenizer.init_kwargs.get('_commit_hash') != metadata['tokenizer_commit']:
        raise ValueError('Model or tokenizer differs from original calibration')
    windows = load_windows(tokenizer, split, count, 128, metadata['dataset_revision'])
    return model, windows


def verify_gate(path):
    gate = json.loads(path.read_text())
    if gate['sources'] != source_hashes():
        raise ValueError('Source files changed; use a new output directory')
    for entry in gate['entries']:
        for artifact, digest in entry['artifacts'].items():
            if file_digest(artifact) != digest:
                raise ValueError(f'Artifact changed: {artifact}')
    return gate


def calibrate(source, output, batch_size, device):
    gate_path = output / 'calibration_complete.json'
    if gate_path.exists():
        verify_gate(gate_path)
        return
    entries = []
    for name, layers in LAYERS.items():
        metadata = json.loads((source / 'descriptive' / f"{name}_{6 if name == 'gpt2' else 16}" / 'metadata.json').read_text())
        model, windows = model_and_windows(name, 'train', 128, metadata, device)
        for layer in layers:
            directory = output / 'calibration' / f'{name}_{layer}'
            directory.mkdir(parents=True, exist_ok=True)
            activation_path = directory / 'activations.npy'
            calibration_path = directory / 'calibration.pt'
            description_path = directory / 'channels.csv'
            removal_path = directory / 'training_removal.csv'
            record_path = directory / 'record.json'
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if record['sources'] != source_hashes() or any(file_digest(p) != h for p, h in record['artifacts'].items()):
                    raise ValueError('Calibration checkpoint changed')
                entries.append(record)
                continue
            activations = collect_activations(model, name, layer, windows, activation_path, batch_size, device)
            norms = outgoing_norms(model, name, layer)
            calibration = fit_calibration(activations, fractions=[.75])
            calibration['metadata'] = {**metadata, 'layer': layer, 'window_count': 128, 'window_size': 128}
            calibration['activation_sha256'] = file_digest(activation_path)
            torch.save(calibration, calibration_path)
            means = calibration['means'].numpy()
            pd.DataFrame({'channel': np.arange(len(means)), 'mean': means,
                          'mean_absolute': np.mean(np.abs(activations), axis=0),
                          'output_norm': norms.numpy()}).to_csv(description_path, index=False)
            removal = []
            for method in METHODS:
                counts = 0
                for offset in range(0, len(activations), 128):
                    values = torch.from_numpy(np.array(activations[offset:offset + 128], copy=True))
                    counts += int(build_mask(values, method, .75, calibration['thresholds'][.75], norms, calibration['means']).sum())
                removal.append({'method': method, 'fraction': .75, 'observed_fraction': counts / activations.size})
            pd.DataFrame(removal).to_csv(removal_path, index=False)
            artifacts = {str(p.resolve()): file_digest(p) for p in (activation_path, calibration_path, description_path, removal_path)}
            record = {'model': name, 'layer': layer, 'sources': source_hashes(), 'artifacts': artifacts,
                      'train_windows': [w.window_id for w in windows], 'scope': 'exploratory'}
            write_json(record_path, record)
            entries.append(record)
            print(f'Calibrated {name} layer {layer}', flush=True)
        del model
    write_json(gate_path, {'sources': source_hashes(), 'entries': entries, 'scope': 'exploratory'})


def run_layer(name, layer, split, count, metadata, calibration_path, directory, batch_size, device):
    model, windows = model_and_windows(name, split, count, metadata, device)
    module = projection(model, name, layer)
    norms = outgoing_norms(model, name, layer).to(device)
    validate_hook(model, module, torch.tensor([windows[0].input_ids], device=device), len(norms))
    calibration = torch.load(calibration_path, map_location='cpu', weights_only=False)
    conditions = [(m, False) for m in METHODS] + [(m, True) for m in METHODS] + [('identity', False), ('all_zero', False)]
    signature = {'model': name, 'layer': layer, 'split': split, 'count': count, 'batch_size': batch_size, 'device': device,
                 'calibration_sha256': file_digest(calibration_path), 'sources': source_hashes(),
                 'windows': [w.window_id for w in windows], 'conditions': conditions}
    signature = json.loads(json.dumps(signature))
    directory.mkdir(parents=True, exist_ok=True)
    signature_path = directory / 'signature.json'
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError('Checkpoint signature changed')
    if not signature_path.exists():
        write_json(signature_path, signature)
    rows = []
    for offset in range(0, count, batch_size):
        batch = windows[offset:offset + batch_size]
        checkpoint = directory / f'batch_{offset:04d}.json'
        if checkpoint.exists():
            batch_rows = json.loads(checkpoint.read_text())
            expected = {(w.window_id, m + ('_random' if random else '')) for w in batch for m, random in conditions}
            if len(batch_rows) != len(expected) or {(r['window_id'], r['method']) for r in batch_rows} != expected:
                raise ValueError('Invalid checkpoint')
            rows.extend(batch_rows)
            continue
        ids = torch.tensor([w.input_ids for w in batch], device=device)
        with torch.inference_mode():
            baseline = model(ids, use_cache=False).logits
        batch_rows = []
        for method, random_control in conditions:
            fraction = 0. if method == 'identity' else 1. if method == 'all_zero' else .75
            actual_method = 'L' if method in ('identity', 'all_zero') else method
            values = evaluate(model, module, ids, batch, name, layer, actual_method, fraction, calibration, norms, random_control, 0, baseline)
            label = method + ('_random' if random_control else '')
            for window, metrics in zip(batch, values):
                if method == 'identity' and abs(metrics['kl']) > 1e-6:
                    raise ValueError('Identity check failed')
                if method == 'all_zero' and metrics['observed_fraction'] != 1:
                    raise ValueError('All-zero check failed')
                batch_rows.append(dict(model=name, layer=layer, depth='extension', window_id=window.window_id,
                                       cluster_id=window.passage_id, method=label, fraction=fraction, repetition=0, **metrics))
        write_json(checkpoint, batch_rows)
        rows.extend(batch_rows)
        print(f'{split}: {name} layer {layer} {min(offset + batch_size, count)}/{count}', flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(directory / 'metrics.csv', index=False)
    table.to_parquet(directory / 'metrics.parquet', index=False)
    paired_summary(rows).to_csv(directory / 'paired_contrasts.csv', index=False)
    record = {'model': name, 'layer': layer, 'split': split, 'window_count': count, 'status': 'passed',
              'calibration_sha256': file_digest(calibration_path), 'sources': source_hashes(), 'scope': 'exploratory'}
    write_json(directory / 'run.json', record)
    del model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['calibrate', 'validation', 'test'])
    parser.add_argument('--source', type=Path, default=ROOT / 'results/study')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/depth_extension')
    parser.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch size must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'calibrate':
        calibrate(args.source, output, args.batch_size, args.device)
        return
    gate = verify_gate(output / 'calibration_complete.json')
    if {(e['model'], e['layer']) for e in gate['entries']} != {(m, l) for m, layers in LAYERS.items() for l in layers}:
        raise ValueError('Incomplete training descriptions')
    if args.stage == 'test':
        validation = verify_gate(output / 'validation_complete.json')
        if {(e['model'], e['layer']) for e in validation['entries']} != set(VALIDATION_LAYERS.items()):
            raise ValueError('Incomplete validation')
    entries = []
    targets = [(name, layer) for name, layers in LAYERS.items() for layer in layers if args.stage == 'test' or layer == VALIDATION_LAYERS[name]]
    for name, layer in targets:
        calibration_path = output / 'calibration' / f'{name}_{layer}' / 'calibration.pt'
        calibration = torch.load(calibration_path, map_location='cpu', weights_only=False)
        directory = output / args.stage / f'{name}_{layer}'
        run_layer(name, layer, args.stage, 4 if args.stage == 'validation' else 128,
                  calibration['metadata'], calibration_path, directory, args.batch_size, args.device)
        artifacts = {str((directory / filename).resolve()): file_digest(directory / filename) for filename in ('signature.json', 'run.json', 'metrics.csv', 'metrics.parquet', 'paired_contrasts.csv')}
        entries.append({'model': name, 'layer': layer, 'artifacts': artifacts})
    write_json(output / f'{args.stage}_complete.json', {'sources': source_hashes(), 'entries': entries, 'scope': 'exploratory'})


if __name__ == '__main__':
    main()
