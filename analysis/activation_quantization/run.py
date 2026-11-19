#!/usr/bin/env python3
"""Calibrate and evaluate SD-band and uniform activation quantization."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from analysis.activation_quantization.quantization import (
    METHODS, ActivationQuantizer, fit_codebooks, quantized_forward,
)
from analysis.large_study.data import DATASET_REVISION, load_sample
from analysis.large_study.evaluate import write_json
from scripts.ablate import MODELS, file_digest
from src.data import load_windows
from src.evaluate import prediction_metrics
from src.model_hooks import projection

REVISIONS = {'gpt2': '607a30d783dfa663caf39e06633721c8d4cfcd7e',
             'smollm2': '93efa2f097d58c2a74874c7e644dbc9b0cee75a2'}
BITS = (4, 6, 8)
MIDDLE = {'gpt2': 6, 'smollm2': 16}
METRICS = ('kl', 'baseline_nll', 'nll_delta', 'flip_rate')


def source_hashes():
    paths = [Path(__file__), Path(__file__).with_name('quantization.py'),
             Path(__file__).with_name('PROTOCOL.md'), ROOT / 'src/data.py',
             ROOT / 'src/evaluate.py', ROOT / 'src/model_hooks.py',
             ROOT / 'analysis/large_study/data.py']
    return {str(path.relative_to(ROOT)): file_digest(path) for path in paths}


def window_records(windows):
    return [{'id': window.window_id, 'article': window.passage_id,
             'input_ids_sha256': hashlib.sha256(json.dumps(window.input_ids, separators=(',', ':')).encode()).hexdigest()}
            for window in windows]


def check_signature(path, signature):
    if path.exists() and json.loads(path.read_text()) != signature:
        raise ValueError(f'Run configuration changed at {path}; choose another output directory')
    if not path.exists():
        write_json(path, signature)


def calibrate(model, modules, windows, model_name, args):
    directory = args.output / model_name
    signature = {'model': model_name, 'model_commit': REVISIONS[model_name],
                 'dataset_revision': DATASET_REVISION, 'split': 'train',
                 'device': args.device, 'batch_size': args.batch_size, 'dtype': 'float32',
                 'windows': window_records(windows), 'bits': list(BITS), 'sample_per_batch': 4096,
                 'source_sha256': source_hashes()}
    path = directory / 'calibration.json'
    if path.exists():
        existing = json.loads(path.read_text())
        if existing['signature'] != signature:
            raise ValueError('Calibration configuration changed; choose another output directory')
        return existing, path
    moments = [dict(count=0, total=0., squared_total=0., maximum=0.) for _ in modules]
    samples = [[] for _ in modules]
    handles = []
    offset = 0
    for layer, module in enumerate(modules):
        def capture(_module, inputs, layer=layer):
            values = inputs[0].detach().float().reshape(-1)
            summary = torch.stack((values.mean(), values.square().mean(), values.abs().max())).cpu().tolist()
            record = moments[layer]
            record['count'] += values.numel()
            record['total'] += summary[0] * values.numel()
            record['squared_total'] += summary[1] * values.numel()
            record['maximum'] = max(record['maximum'], summary[2])
            generator = torch.Generator(device='cpu').manual_seed(104729 + 1009 * layer + offset)
            indices = torch.randint(values.numel(), (min(4096, values.numel()),), generator=generator)
            samples[layer].append(values[indices.to(values.device)].cpu().numpy())
        handles.append(module.register_forward_pre_hook(capture))
    try:
        with torch.inference_mode():
            for offset in range(0, len(windows), args.batch_size):
                ids = torch.tensor([window.input_ids for window in windows[offset:offset + args.batch_size]], device=args.device)
                model(ids, use_cache=False)
                print(f'{model_name}: calibration {min(offset + args.batch_size, len(windows))}/{len(windows)}', flush=True)
    finally:
        for handle in handles:
            handle.remove()
    layers, candidates, retained_samples = {}, [], {}
    for layer, record in enumerate(moments):
        mean = record['total'] / record['count']
        sigma = float(np.sqrt(max(0., record['squared_total'] / record['count'] - mean ** 2)))
        sample = np.concatenate(samples[layer])
        if not np.isfinite(sample).all() or sigma <= 0:
            raise ValueError(f'Invalid training activations in layer {layer}')
        retained_samples[f'layer_{layer}'] = sample
        codebooks = {}
        for bits in BITS:
            fitted, search = fit_codebooks(sample, sigma, record['maximum'], bits)
            codebooks[str(bits)] = fitted
            candidates.extend(dict(model=model_name, layer=layer, bits=bits, **item) for item in search)
        layers[str(layer)] = {'mean': mean, 'sigma': sigma, 'maximum': record['maximum'],
                              'activation_count': record['count'], 'sample_count': len(sample),
                              'codebooks': codebooks}
        print(f'{model_name}: fitted layer {layer}', flush=True)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / 'calibration_samples.npz', **retained_samples)
    calibration = {'signature': signature, 'layers': layers,
                   'samples_sha256': file_digest(directory / 'calibration_samples.npz')}
    pd.DataFrame(candidates).to_csv(directory / 'clipping_search.csv', index=False)
    write_json(path, calibration)
    return calibration, path


def evaluate(model, modules, windows, calibration, calibration_path, model_name, stage, args, manifest):
    directory = args.output / model_name / stage
    directory.mkdir(parents=True, exist_ok=True)
    signature = {'model': model_name, 'model_commit': REVISIONS[model_name], 'stage': stage,
                 'device': args.device, 'batch_size': args.batch_size, 'dtype': 'float32',
                 'windows': window_records(windows), 'manifest': manifest,
                 'calibration_sha256': file_digest(calibration_path), 'source_sha256': source_hashes()}
    check_signature(directory / 'signature.json', signature)
    scopes = {'middle': [MIDDLE[model_name]], 'all_layers': list(range(len(modules)))}
    quantizers = {(bits, method): [ActivationQuantizer(calibration['layers'][str(layer)]['codebooks'][str(bits)][method]['levels'], args.device)
                                  for layer in range(len(modules))]
                  for bits in BITS for method in METHODS}
    expected_conditions = {(scope, bits, method) for scope in scopes for bits in BITS for method in METHODS}
    rows = []
    with torch.inference_mode():
        for offset in range(0, len(windows), args.batch_size):
            batch = windows[offset:offset + args.batch_size]
            checkpoint = directory / f'batch_{offset:05d}.json'
            expected = {(window.window_id, *condition) for window in batch for condition in expected_conditions}
            if checkpoint.exists():
                saved = json.loads(checkpoint.read_text())
                actual = {(row['window_id'], row['scope'], row['bits'], row['method']) for row in saved}
                if len(saved) != len(expected) or actual != expected or any(not np.isfinite(row[key]) for row in saved for key in METRICS):
                    raise ValueError('Incomplete or non-finite evaluation checkpoint')
                rows.extend(saved)
                continue
            started = time.monotonic()
            ids = torch.tensor([window.input_ids for window in batch], device=args.device)
            baseline = model(ids, use_cache=False).logits
            if stage == 'validation':
                for indices in scopes.values():
                    identity = quantized_forward(model, [modules[layer] for layer in indices], [lambda value: value] * len(indices), ids)
                    if not torch.allclose(baseline, identity, rtol=1e-5, atol=1e-6):
                        raise ValueError('Identity hooks changed model logits')
                    del identity
            batch_rows = []
            for scope, indices in scopes.items():
                for bits in BITS:
                    for method in METHODS:
                        changed = quantized_forward(model, [modules[layer] for layer in indices],
                                                    [quantizers[(bits, method)][layer] for layer in indices], ids)
                        metrics = prediction_metrics(baseline, changed, ids)
                        del changed
                        arrays = {key: value.cpu().tolist() for key, value in metrics.items()}
                        for index, window in enumerate(batch):
                            values = {key: arrays[key][index] for key in METRICS}
                            if not all(np.isfinite(list(values.values()))):
                                raise ValueError('Non-finite prediction metric')
                            batch_rows.append(dict(model=model_name, scope=scope, bits=bits, method=method,
                                                   window_id=window.window_id, cluster_id=window.passage_id, **values))
            del baseline
            write_json(checkpoint, batch_rows)
            rows.extend(batch_rows)
            print(f'{model_name}: {stage} {min(offset + len(batch), len(windows))}/{len(windows)} ({time.monotonic() - started:.1f}s)', flush=True)
    pd.DataFrame(rows).to_csv(directory / 'metrics.csv', index=False)
    write_json(directory / 'run.json', {'status': 'passed', 'model': model_name, 'stage': stage,
                                        'windows': len(windows), 'articles': len({window.passage_id for window in windows}),
                                        'signature_sha256': file_digest(directory / 'signature.json'),
                                        'identity_hook_parity': stage == 'validation'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', choices=MODELS, nargs='+', default=list(MODELS))
    parser.add_argument('--count', type=int, default=64)
    parser.add_argument('--calibration-count', type=int, default=128)
    parser.add_argument('--validation-count', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/activation_quantization')
    args = parser.parse_args()
    if min(args.count, args.calibration_count, args.validation_count, args.batch_size) < 1 or len(set(args.models)) != len(args.models):
        parser.error('counts must be positive and models distinct')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizers = {name: AutoTokenizer.from_pretrained(identifier, revision=REVISIONS[name], local_files_only=True)
                  for name, identifier in MODELS.items()}
    windows_by_model, manifest = load_sample(tokenizers, count=args.count)
    original = json.loads((ROOT / 'results/large_study/sample_manifest.json').read_text())
    if manifest['dataset_text_sha256'] != original['dataset_text_sha256']:
        raise ValueError('Held-out dataset text differs from the original study')
    for name in MODELS:
        if manifest['models'][name]['windows'] != original['models'][name]['windows'][:args.count]:
            raise ValueError('Held-out token windows differ from the original study')
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'sample_manifest.json', manifest)
    for name in args.models:
        model = AutoModelForCausalLM.from_pretrained(MODELS[name], revision=REVISIONS[name],
                                                    local_files_only=True, torch_dtype=torch.float32).to(args.device).eval()
        if model.config._commit_hash != REVISIONS[name]:
            raise ValueError('Model commit differs from the original study')
        count = model.config.n_layer if name == 'gpt2' else model.config.num_hidden_layers
        modules = [projection(model, name, layer) for layer in range(count)]
        training = load_windows(tokenizers[name], 'train', args.calibration_count, 128, DATASET_REVISION)
        calibration, path = calibrate(model, modules, training, name, args)
        validation = load_windows(tokenizers[name], 'validation', args.validation_count, 128, DATASET_REVISION)
        evaluate(model, modules, validation, calibration, path, name, 'validation', args, {'split': 'validation'})
        evaluate(model, modules, windows_by_model[name], calibration, path, name, 'test', args, manifest)
        del modules, model
        gc.collect()
        if args.device == 'mps':
            torch.mps.empty_cache()


if __name__ == '__main__':
    main()
