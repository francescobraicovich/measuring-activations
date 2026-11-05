"""Describe training selection and overlap for the expanded calibration."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from analysis.expanded.methods import FRACTIONS, METHODS
from analysis.expanded.run import LAYERS, file_digest


def describe(values, norms, calibration, batch_size=512):
    means = calibration['means'].numpy()
    totals = {(method, p): np.zeros(4, dtype=np.float64) for method in METHODS for p in FRACTIONS}
    overlaps = {method: np.zeros(3, dtype=np.int64) for method in ('N', 'L', 'G', 'W')}
    examples = []
    for start in range(0, len(values), batch_size):
        a = torch.from_numpy(np.array(values[start:start + batch_size], copy=True))
        raw = a.abs().numpy()
        centered = (a - calibration['means']).abs().numpy()
        weighted = raw * norms
        ranks = {}
        for method, scores in (("L", raw), ("W", weighted), ("L_centered", centered),
                               ("W_centered", centered * norms)):
            order = np.argsort(scores, axis=-1, kind="stable")
            ranks[method] = np.argsort(order, axis=-1)
        for p in FRACTIONS:
            masks = {}
            for method in METHODS:
                if p == 0:
                    mask = np.zeros_like(raw, dtype=bool)
                elif p == 1:
                    mask = np.ones_like(raw, dtype=bool)
                elif method in ranks:
                    mask = ranks[method] < int(p * raw.shape[-1])
                else:
                    scores = centered if method.endswith('_centered') else raw
                    mask = scores < calibration['thresholds'][p][method].numpy()
                masks[method] = mask
                totals[method, p] += [mask.sum(), (raw * mask).sum(dtype=np.float64),
                                      (weighted * mask).sum(dtype=np.float64), (centered * mask).sum(dtype=np.float64)]
            if p == .75:
                for method in overlaps:
                    left, right = masks[method], masks[method + '_centered']
                    overlaps[method] += [np.logical_and(left, right).sum(), np.logical_or(left, right).sum(), np.logical_xor(left, right).sum()]
                if start == 0:
                    for method in overlaps:
                        examples.append({'token_index': 0, 'method': method,
                                         'raw_selected_count': int(masks[method][0].sum()),
                                         'centered_selected_count': int(masks[method + '_centered'][0].sum()),
                                         'intersection_count': int((masks[method][0] & masks[method + '_centered'][0]).sum())})
    count = values.size
    rows = []
    for (method, p), (removed, raw, weighted, centered) in totals.items():
        rows.append({'method': method, 'fraction': p, 'removal_rate': removed / count,
                     'raw_removed_mass_per_event': raw / count, 'weighted_removed_mass_per_event': weighted / count,
                     'centered_removed_mass_per_event': centered / count})
    comparisons = {method: {'jaccard': int(intersection) / int(union) if union else 1.0,
                            'disagreement_rate': int(difference) / count}
                   for method, (intersection, union, difference) in overlaps.items()}
    return {'selection': rows, 'p75_overlap': comparisons, 'examples': examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results/study/descriptive'))
    parser.add_argument('--calibration', type=Path, default=Path('results/expanded/calibration'))
    parser.add_argument('--output', type=Path, default=Path('results/expanded/training_description.json'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'output already exists: {args.output}')
    gate = json.loads((args.calibration / 'train_complete.json').read_text())
    entries = []
    for model, layers in LAYERS.items():
        for layer in layers:
            label = f'{model}_{layer}'
            path = args.calibration / f'{label}.pt'
            matching = [entry for entry in gate['entries'] if entry['model'] == model and entry['layer'] == layer]
            if len(matching) != 1 or matching[0]['sha256'] != file_digest(path):
                raise ValueError(f'calibration integrity check failed for {label}')
            calibration = torch.load(path, map_location='cpu', weights_only=False)
            source = args.source / label / 'activations.npy'
            if calibration['activation_sha256'] != file_digest(source):
                raise ValueError(f'training activation integrity check failed for {label}')
            values = np.load(source, mmap_mode='r')
            norms = np.load(args.source / label / 'outgoing_norms.npy')
            description = describe(values, norms, calibration)
            description.update({'model': model, 'layer': layer, 'calibration_sha256': file_digest(path),
                                'activation_sha256': calibration['activation_sha256'],
                                'mean_absolute_channel_mean': float(calibration['means'].abs().mean()),
                                'max_absolute_channel_mean': float(calibration['means'].abs().max()),
                                'threshold_ties': {str(p): {method: {'mean': float(np.mean(rate)), 'max': float(np.max(rate))}
                                                             for method, rate in rates.items()}
                                                   for p, rates in calibration['tie_fractions'].items()}})
            entries.append(description)
            print(f'Described {label}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'split': 'train', 'scope': 'exploratory', 'entries': entries}, indent=2) + '\n')


if __name__ == '__main__':
    main()
