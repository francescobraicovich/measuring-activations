#!/usr/bin/env python3
"""Supplement held-out results with token quotas, threshold ties and pooled masses.

The frozen model is run without interventions. All thresholds come from the
existing train calibration; no test data is used to fit a quantity.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.calibrate import load_calibration
from src.data import load_windows
from src.masks import build_mask
from src.model_hooks import outgoing_norms, projection

MODELS = {'gpt2': ('openai-community/gpt2', (2, 6, 11)),
          'smollm2': ('HuggingFaceTB/SmolLM2-135M', (5, 16, 29))}


def diagnose(model, tokenizer, name, layer, device, batch_size):
    root = ROOT / 'results/study'
    calibration = load_calibration(root / 'descriptive' / f'{name}_{layer}' / 'calibration.pt')
    metadata = calibration['metadata']
    if metadata['model_commit'] != model.config._commit_hash or metadata['layer'] != layer:
        raise ValueError('Model or layer differs from the train calibration')
    windows = load_windows(tokenizer, 'test', 128, 128, metadata['dataset_revision'])
    module = projection(model, name, layer)
    norms = outgoing_norms(model, name, layer).cpu()
    fractions = [0., .25, .5, .75, .9, 1.] if layer in (6, 16) else [.75]
    per_token, per_window, summaries, ties = [], [], [], []
    aggregates = {(method, p): np.zeros(5, dtype=np.float64)
                  for method in 'NLGW' for p in fractions}
    tie_counts = {(method, p): np.zeros(len(norms) if method == 'N' else 1, dtype=np.int64)
                  for method in 'NG' for p in fractions}
    total_abs = total_squared = total_weighted = 0.0
    total_events = total_tokens = 0
    captured = []
    def capture(_module, inputs):
        captured.append(inputs[0].detach().cpu())
    handle = module.register_forward_pre_hook(capture)
    try:
        for offset in range(0, len(windows), batch_size):
            batch = windows[offset:offset+batch_size]
            ids = torch.tensor([window.input_ids for window in batch], device=device)
            with torch.inference_mode():
                model(ids, use_cache=False)
            if len(captured) != 1:
                raise RuntimeError('Projection hook did not capture exactly one batch')
            activation = captured.pop()
            if tuple(activation.shape) != (len(batch), 128, len(norms)):
                raise ValueError(f'Unexpected activation shape {activation.shape}')
            absolute = activation.abs()
            weighted = absolute * norms
            squared = absolute.square()
            total_abs += absolute.sum(dtype=torch.float64).item()
            total_squared += squared.sum(dtype=torch.float64).item()
            total_weighted += weighted.sum(dtype=torch.float64).item()
            total_events += absolute.numel()
            total_tokens += len(batch) * 128
            for p in fractions:
                thresholds = calibration['thresholds'][p]
                for method in 'NLGW':
                    mask = build_mask(activation, method, p, thresholds, norms)
                    counts = mask.sum(-1)
                    if method in 'LW' and not torch.all(counts == int(p * len(norms))):
                        raise RuntimeError('L/W per-token counts are not exact')
                    values = aggregates[(method, p)]
                    values += [mask.sum().item(), (absolute * mask).sum(dtype=torch.float64).item(),
                               (squared * mask).sum(dtype=torch.float64).item(),
                               (weighted * mask).sum(dtype=torch.float64).item(), mask.numel()]
                    for i, window in enumerate(batch):
                        per_window.append({'model': name, 'layer': layer, 'fraction': p, 'method': method,
                                           'window_id': window.window_id,
                                           'removed_fraction': float(counts[i].sum()) / mask[i].numel()})
                    if method in 'NG':
                        threshold = thresholds[method]
                        tied = absolute == threshold
                        if method == 'N':
                            tie_counts[(method,p)] += tied.sum((0,1)).numpy()
                        else:
                            tie_counts[(method,p)][0] += tied.sum().item()
                        for i, window in enumerate(batch):
                            for position, count in enumerate(counts[i].tolist()):
                                per_token.append({'model': name, 'layer': layer, 'fraction': p,
                                                  'method': method, 'window_id': window.window_id,
                                                  'token_position': position, 'removed_count': count,
                                                  'channel_count': len(norms)})
            if offset % 16 == 0:
                print(f'{name} layer {layer}: {offset + len(batch)}/128 windows', flush=True)
    finally:
        handle.remove()
    recorded = pd.read_parquet(root / 'test' / f'{name}_{layer}' / 'metrics.parquet')
    recorded = recorded[recorded.method.isin(list('NLGW'))]
    window_table = pd.DataFrame(per_window)
    window_table = window_table[window_table.fraction.isin([.25, .5, .75, .9])]
    observed = window_table.merge(recorded[['window_id','method','fraction','observed_fraction']],
                  on=['window_id','method','fraction'], validate='one_to_one')
    if len(observed) != len(window_table) or len(observed) != len(recorded):
        raise ValueError('Held-out and saved result conditions do not match')
    error = np.max(np.abs(observed.removed_fraction - observed.observed_fraction))
    if error > 1e-6:
        raise RuntimeError(f'Held-out mask counts differ from intervention run: {error}')
    for (method, p), values in aggregates.items():
        summaries.append({'model': name, 'layer': layer, 'fraction': p, 'method': method,
                          'events': total_events, 'tokens': total_tokens,
                          'removed_fraction': values[0]/total_events,
                          'removed_abs_fraction': values[1]/total_abs,
                          'removed_squared_fraction': values[2]/total_squared,
                          'removed_weighted_fraction': values[3]/total_weighted,
                          'max_window_count_difference': error})
    for (method,p), counts in tie_counts.items():
        if method == 'N':
            ties.extend({'model': name, 'layer': layer, 'fraction': p, 'method': method,
                         'channel': i, 'tie_fraction': int(count)/total_tokens}
                        for i,count in enumerate(counts))
        else:
            ties.append({'model': name, 'layer': layer, 'fraction': p, 'method': method,
                         'channel': -1, 'tie_fraction': int(counts[0])/total_events})
    output = root / 'heldout_diagnostics' / f'{name}_{layer}'
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output / 'pooled_mass.csv', index=False)
    pd.DataFrame(ties).to_csv(output / 'threshold_ties.csv', index=False)
    pd.DataFrame(per_token).to_parquet(output / 'token_removal.parquet', index=False)
    print(f'Completed {name} layer {layer}, count discrepancy {error:.2g}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    for name, (model_id, layers) in MODELS.items():
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(args.device).eval()
        for layer in layers:
            diagnose(model, tokenizer, name, layer, args.device, args.batch_size)
        del model

if __name__ == '__main__':
    main()
