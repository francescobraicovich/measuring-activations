#!/usr/bin/env python3
"""Summarize exploratory paired comparisons from saved per-window metrics."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('N', 'L', 'G', 'W')
COLUMNS = ('kl', 'observed_fraction', 'removed_abs_fraction', 'removed_centered_abs_fraction')


def prepare(frame, method, fraction):
    values = frame[(frame.method == method) & np.isclose(frame.fraction, fraction)].copy()
    if values.empty or values.duplicated(['window_id', 'repetition']).any():
        raise ValueError(f'Missing or duplicate observations for {method} at {fraction}')
    if values[list(COLUMNS)].isna().any().any() or not np.isfinite(values[list(COLUMNS)].to_numpy()).all():
        raise ValueError(f'Non-finite observations for {method} at {fraction}')
    if not values.groupby('window_id').cluster_id.nunique().eq(1).all():
        raise ValueError('Window has inconsistent article identifiers')
    repetitions = values.groupby('window_id').repetition.agg(lambda x: tuple(sorted(x)))
    if repetitions.nunique() != 1:
        raise ValueError(f'Inconsistent repetitions for {method} at {fraction}')
    return values.groupby(['window_id', 'cluster_id'], as_index=False)[list(COLUMNS)].mean()


def paired(frame, first, second, fraction, replicates=1000):
    left = prepare(frame, first, fraction)
    right = prepare(frame, second, fraction)
    if set(left.window_id) != set(right.window_id):
        raise ValueError(f'Unmatched windows: {first} versus {second} at {fraction}')
    values = left.merge(right, on='window_id', suffixes=('_first', '_second'), validate='one_to_one')
    if not values.cluster_id_first.eq(values.cluster_id_second).all():
        raise ValueError('Paired windows disagree on article identifier')
    values['difference'] = values.kl_first - values.kl_second
    clusters = values.groupby('cluster_id_first').difference.agg(['sum', 'count'])
    rng = np.random.default_rng(0)
    samples = rng.integers(0, len(clusters), size=(replicates, len(clusters)))
    estimates = clusters['sum'].to_numpy()[samples].sum(axis=1) / clusters['count'].to_numpy()[samples].sum(axis=1)
    result = {'first': first, 'second': second, 'fraction': fraction, 'windows': len(values),
              'articles': len(clusters), 'replicates': replicates, 'paired_kl_difference': values.difference.mean(),
              'lower': np.quantile(estimates, .025), 'upper': np.quantile(estimates, .975)}
    for column in COLUMNS:
        for side in ('first', 'second'):
            result[f'{side}_{column}'] = values[f'{column}_{side}'].mean()
    return result


def analyze(frame):
    required = {'model', 'layer', 'window_id', 'cluster_id', 'method', 'fraction', 'repetition', *COLUMNS}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(frame.columns))}')
    if frame[['model', 'layer', 'window_id', 'cluster_id', 'method', 'fraction', 'repetition']].isna().any().any():
        raise ValueError('Missing identifiers')
    if not np.isfinite(frame[['fraction', 'repetition']].to_numpy()).all():
        raise ValueError('Non-finite fraction or repetition')
    results = []
    for (model, layer), group in frame.groupby(['model', 'layer']):
        fractions = sorted(group.loc[group.method == 'L', 'fraction'].unique())
        if not fractions:
            raise ValueError(f'No L observations for {model} layer {layer}')
        for fraction in fractions:
            comparisons = [(m + '_centered', m) for m in METHODS]
            comparisons += [('W', 'L'), ('W_centered', 'L_centered')]
            comparisons += [(m, m + '_random') for m in (*METHODS, *(m + '_centered' for m in METHODS))]
            for first, second in comparisons:
                results.append({'model': model, 'layer': layer, **paired(group, first, second, fraction)})
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'results/expanded/test')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/expanded/analysis')
    args = parser.parse_args()
    paths = sorted(args.input.glob('*/metrics.csv'))
    if not paths:
        raise ValueError('No saved metrics found')
    result = analyze(pd.concat([pd.read_csv(path) for path in paths], ignore_index=True))
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / 'paired_comparisons.csv', index=False)
    lines = ['# Exploratory paired comparisons', '',
             'Differences are first minus second in KL nats per scored token. Percentile intervals use 1,000 paired article-cluster bootstrap replicates, weighted by window counts; random repetitions are averaged within each window.',
             'Four test articles per model make intervals fragile. Centered scores select events, but selected activations are still set to zero. Removed mass values are means of per-window fractions of raw or centered absolute mass, across all 128 positions.', '']
    for (model, layer), group in result.groupby(['model', 'layer']):
        lines += [f'## {model}, layer {layer}', '', '| Fraction | First − second | KL difference [95% interval] | Observed removal first / second | Raw mass first / second | Centered mass first / second |',
                  '|---:|---|---:|---:|---:|---:|']
        for row in group.itertuples():
            lines.append(f'| {row.fraction:.2f} | {row.first} − {row.second} | {row.paired_kl_difference:.5f} [{row.lower:.5f}, {row.upper:.5f}] | {row.first_observed_fraction:.3f} / {row.second_observed_fraction:.3f} | {row.first_removed_abs_fraction:.3f} / {row.second_removed_abs_fraction:.3f} | {row.first_removed_centered_abs_fraction:.3f} / {row.second_removed_centered_abs_fraction:.3f} |')
        lines.append('')
    (args.output / 'paired_summary.md').write_text('\n'.join(lines) + '\n')
    print(f'Wrote {len(result)} paired comparisons to {args.output}')


if __name__ == '__main__':
    main()
