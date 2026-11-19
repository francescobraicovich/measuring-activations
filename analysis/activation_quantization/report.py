#!/usr/bin/env python3
"""Summarize the paired activation quantization experiment and export figures."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from analysis.activation_quantization.quantization import METHODS

METRICS = ('kl', 'baseline_nll', 'nll_delta', 'flip_rate')
LABELS = {'uniform_max': 'Uniform · training maximum', 'uniform_mse': 'Uniform · MSE-selected range',
          'g_geometric': 'G · increasing bin density', 'uniform_matched': 'Uniform · G-matched range'}


def article_intervals(frame, metrics, seed=0, replicates=1000):
    """Resample whole articles and retain the per-window weighting."""
    clusters = frame.groupby('cluster_id')[list(metrics)].sum()
    counts = frame.groupby('cluster_id').size().reindex(clusters.index).to_numpy()
    indices = np.random.default_rng(seed).integers(0, len(clusters), size=(replicates, len(clusters)))
    estimates = clusters.to_numpy()[indices].sum(axis=1) / counts[indices].sum(axis=1, keepdims=True)
    intervals = np.quantile(estimates, [.025, .975], axis=0)
    return {metric: (float(frame[metric].mean()), float(intervals[0, index]), float(intervals[1, index]))
            for index, metric in enumerate(metrics)}


def summarize(frame):
    records = []
    for keys, group in frame.groupby(['model', 'scope', 'bits', 'method'], sort=True):
        record = dict(zip(('model', 'scope', 'bits', 'method'), keys))
        for metric, values in article_intervals(group, METRICS).items():
            record.update({f'{metric}_{name}': value for name, value in zip(('mean', 'lower', 'upper'), values)})
        record.update(windows=len(group), articles=group.cluster_id.nunique(),
                      baseline_ppl=math.exp(record['baseline_nll_mean']),
                      ppl=math.exp(record['baseline_nll_mean'] + record['nll_delta_mean']))
        records.append(record)
    return pd.DataFrame(records)


def paired_comparisons(frame):
    records = []
    for keys, group in frame.groupby(['model', 'scope', 'bits'], sort=True):
        left = group[group.method == 'g_geometric']
        for reference in ('uniform_max', 'uniform_mse', 'uniform_matched'):
            right = group[group.method == reference]
            paired = left.merge(right, on=['window_id', 'cluster_id'], suffixes=('_left', '_right'), validate='one_to_one')
            if left.empty or len(paired) != len(left) or len(paired) != len(right):
                raise ValueError('Paired methods must contain identical windows and articles')
            metrics = ('kl', 'nll_delta', 'flip_rate')
            for metric in metrics:
                paired[metric] = paired[metric + '_left'] - paired[metric + '_right']
            for metric, (mean, lower, upper) in article_intervals(paired, metrics, seed=17).items():
                records.append(dict(zip(('model', 'scope', 'bits'), keys), reference=reference,
                                    metric=metric, mean=mean, lower=lower, upper=upper,
                                    windows=len(paired), articles=paired.cluster_id.nunique()))
    return pd.DataFrame(records)


def calibration_diagnostics(output):
    records = []
    for model in ('gpt2', 'smollm2'):
        calibration = json.loads((output / model / 'calibration.json').read_text())
        with np.load(output / model / 'calibration_samples.npz') as samples:
            for layer, values in calibration['layers'].items():
                magnitude = np.abs(samples[f'layer_{layer}'])
                for bits, books in values['codebooks'].items():
                    for method, book in books.items():
                        records.append(dict(model=model, layer=int(layer), bits=int(bits), method=method,
                                            sigma=values['sigma'], clip_sd=book['clip_sd'],
                                            positive_levels=len(book['levels']) - 1,
                                            positive_levels_within_sd=sum(0 < value <= values['sigma'] for value in book['levels']),
                                            sample_fraction_within_sd=float((magnitude <= values['sigma']).mean()),
                                            sample_zero_fraction=float((magnitude <= book['levels'][1] / 2).mean()),
                                            sample_clip_fraction=book['calibration_clip_fraction'],
                                            calibration_mse=book['calibration_mse']))
    return pd.DataFrame(records)


def export_figures(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'uniform_max': '#777777', 'uniform_mse': '#3264a0',
              'g_geometric': '#b65b29', 'uniform_matched': '#43896c'}
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    for scope in ('all_layers', 'middle'):
        figure, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
        for row, model in enumerate(('gpt2', 'smollm2')):
            selected = summary[(summary.model == model) & (summary.scope == scope)]
            for column, metric in enumerate(('kl', 'nll_delta')):
                axis = axes[row, column]
                for method in METHODS:
                    values = selected[selected.method == method].sort_values('bits')
                    axis.plot(values.bits, values[f'{metric}_mean'], marker='o', markersize=4,
                              color=colors[method], label=LABELS[method], linewidth=1.7,
                              linestyle='--' if method == 'uniform_matched' else '-')
                    axis.fill_between(values.bits.to_numpy(), values[f'{metric}_lower'].to_numpy(),
                                      values[f'{metric}_upper'].to_numpy(), color=colors[method], alpha=.1)
                axis.set_xticks([4, 6, 8])
                axis.set_xlabel('Activation precision (bits)')
                axis.set_ylabel('KL from intact model (nats/token)' if metric == 'kl' else 'NLL change (nats/token)')
                axis.set_title(('GPT-2' if model == 'gpt2' else 'SmolLM2-135M') + (' · all MLP layers' if scope == 'all_layers' else ' · middle MLP layer'))
                axis.grid(alpha=.18)
                if metric == 'kl':
                    axis.set_yscale('log')
                else:
                    axis.axhline(0, color='#555555', linewidth=.7)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, .01))
        figure.tight_layout(rect=(0, .1, 1, 1))
        figure.savefig(output / f'{scope}.png', dpi=180)
        figure.savefig(output / f'{scope}.svg')
        plt.close(figure)


def write_report(summary, paired, diagnostics, output):
    all_layer_kl = paired[(paired.scope == 'all_layers') & (paired.metric == 'kl')]
    against_mse = all_layer_kl[all_layer_kl.reference == 'uniform_mse']
    against_max = all_layer_kl[all_layer_kl.reference == 'uniform_max']
    against_matched = all_layer_kl[all_layer_kl.reference == 'uniform_matched']
    lines = ['# Increasing-density activation quantization', '',
             f'With every MLP layer quantized, G had higher KL than MSE-calibrated uniform '
             f'quantization in {int((against_mse["mean"] > 0).sum())}/{len(against_mse)} model/precision '
             f'conditions. The paired KL intervals were entirely above zero in '
             f'{int((against_mse["lower"] > 0).sum())}/{len(against_mse)} conditions.', '',
             f'G beat naive training-maximum uniform quantization in '
             f'{int((against_max["mean"] < 0).sum())}/{len(against_max)} conditions. However, '
             f'uniform quantization using exactly G\'s clipping range also had lower KL in '
             f'{int((against_matched["mean"] > 0).sum())}/{len(against_matched)} conditions. '
             'Thus the losses cannot be attributed only to different clipping ranges.', '',
             'Exploratory simulated activation quantization in frozen GPT-2 and SmolLM2-135M. '
             'All weights remain float32. Each condition uses 64 nonoverlapping 128-token test windows '
             'across 61 articles, with 111 next-token positions scored per window.', '',
             'G denotes a layer-pooled training reference. SD is measured on signed activations; '
             'bins retain zero as the origin and preserve signs. Quantization interval density '
             'doubles in each successive SD-wide magnitude band. Ratios are normalized to the '
             'bit budget and integer counts per band are approximate. Every method uses 15, 63, '
             'or 255 distinct values at 4, 6, or 8 bits.', '',
             'Clipping is selected separately by layer and precision using only training activation '
             'MSE. Uniform MSE uses the same candidate ranges; uniform matched uses exactly the '
             'range selected for G. Uniform maximum is the ordinary unclipped calibration-range baseline. '
             'No held-out outcome selects parameters.', '',
             'Lower KL, NLL change, and flip rate are preferable. Perplexity is calculated only over '
             'the scored short-window positions, rather than a standard full-context WikiText evaluation.', '']
    lines.extend(['## Allocation diagnostic', '',
                  'The exponential schedule concentrates many levels in the tails. At the middle '
                  'layer with 8-bit G quantization:', ''])
    for model, layer in [('gpt2', 6), ('smollm2', 16)]:
        row = diagnostics[(diagnostics.model == model) & (diagnostics.layer == layer)
                          & (diagnostics.bits == 8) & (diagnostics.method == 'g_geometric')].iloc[0]
        lines.append(f'- {model}: {row.sample_fraction_within_sd:.1%} of sampled training activations '
                     f'lie within one SD of zero, but only {row.positive_levels_within_sd} of '
                     f'{row.positive_levels} positive levels lie in 0--1 SD. The selected range '
                     f'is {row.clip_sd:g} SD.')
    lines.extend(['', 'This suggests that doubling density per SD is too aggressive for these '
                  'distributions. It does not establish that all nonuniform quantizers fail. '
                  'See `calibration_diagnostics.csv` for every layer and method.', ''])
    for scope, heading in [('all_layers', 'All MLP layers together'), ('middle', 'Middle MLP layer alone')]:
        lines.extend([f'## {heading}', '',
                      '| Model | Bits | Method | KL | NLL change | Perplexity | Top-token flips |',
                      '|---|---:|---|---:|---:|---:|---:|'])
        selected = summary[summary.scope == scope]
        for row in selected.itertuples():
            lines.append(f'| {row.model} | {row.bits} | {row.method} | {row.kl_mean:.6g} | {row.nll_delta_mean:+.6g} | {row.ppl:.4f} | {100 * row.flip_rate_mean:.2f}% |')
        lines.extend(['', f'![{heading}]({scope}.png)', '',
                      'Shaded bands are 95% article-cluster bootstrap intervals (1,000 replicates).', ''])
    lines.extend(['## Paired differences against the stronger uniform baseline', '',
                  'G minus uniform with MSE-selected clipping; negative values favor G. '
                  'Intervals are 95% article-cluster bootstrap intervals with 1,000 replicates.', '',
                  '| Model | Scope | Bits | Outcome | Difference | 95% interval |',
                  '|---|---|---:|---|---:|---|'])
    for row in paired[paired.reference == 'uniform_mse'].itertuples():
        lines.append(f'| {row.model} | {row.scope} | {row.bits} | {row.metric} | {row.mean:+.6g} | [{row.lower:+.6g}, {row.upper:+.6g}] |')
    lines.extend(['', '## Limits', '',
                  'Results cover two small models, one corpus, a fixed calibration sample, and the '
                  'MLP output-projection inputs. Attention inputs and the KV cache are not quantized. '
                  'Intervals cover article variation, not calibration or domain variation. '
                  'Activation-MSE clipping is a conventional calibration rule; it may not minimize '
                  'downstream loss. Other bin-density schedules remain untested.', '',
                  'These interventions reconstruct float32 values and run the original model. '
                  'They establish accuracy effects, with no measured memory reduction or speedup. '
                  'Nonuniform levels would require suitable encoding and kernels for efficient deployment.', '',
                  'Reproduce with `analysis/activation_quantization/run.py --device mps` followed by '
                  '`analysis/activation_quantization/report.py`. Use the pinned dependencies and a '
                  'populated Hugging Face cache; see the protocol for all settings.', ''])
    (output / 'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/activation_quantization')
    args = parser.parse_args()
    frames = []
    for model in ('gpt2', 'smollm2'):
        directory = args.output / model / 'test'
        record = json.loads((directory / 'run.json').read_text())
        if record['status'] != 'passed' or record['windows'] != 64 or record['articles'] != 61:
            raise ValueError('Expected the complete 64-window, 61-article experiment')
        frame = pd.read_csv(directory / 'metrics.csv')
        if (len(frame) != 64 * 2 * 3 * len(METHODS)
                or frame.duplicated(['window_id', 'scope', 'bits', 'method']).any()
                or not np.isfinite(frame[list(METRICS)]).all().all()):
            raise ValueError('Incomplete or invalid per-window results')
        frames.append(frame)
    measurements = pd.concat(frames, ignore_index=True)
    summary = summarize(measurements)
    paired = paired_comparisons(measurements)
    diagnostics = calibration_diagnostics(args.output)
    measurements.to_csv(args.output / 'metrics.csv', index=False)
    summary.to_csv(args.output / 'summary.csv', index=False)
    paired.to_csv(args.output / 'paired_comparisons.csv', index=False)
    diagnostics.to_csv(args.output / 'calibration_diagnostics.csv', index=False)
    export_figures(summary, args.output)
    write_report(summary, paired, diagnostics, args.output)
    print(summary[summary.scope == 'all_layers'][['model', 'bits', 'method', 'kl_mean', 'nll_delta_mean', 'ppl']].to_string(index=False))


if __name__ == '__main__':
    main()
