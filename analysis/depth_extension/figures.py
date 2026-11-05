#!/usr/bin/env python3
"""Plot exploratory depth comparisons from saved per-window metrics."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.expanded.paired_analysis import prepare

MODEL_LAYERS = {'gpt2': (12, (0, 2, 4, 6, 8, 10, 11)),
                'smollm2': (30, (0, 5, 10, 16, 21, 26, 29))}
ORIGINAL = {'gpt2': {2, 6, 11}, 'smollm2': {5, 16, 29}}
COLORS = {'L': '#D55E00', 'W': '#CC79A7', 'L_centered': '#0072B2', 'W_centered': '#009E73'}


def summarize(table, method):
    values = prepare(table, method, .75)
    clusters = values.groupby('cluster_id').kl.agg(['sum', 'count'])
    if len(clusters) != 4:
        raise ValueError(f'Expected four test articles for {method}, got {len(clusters)}')
    rng = np.random.default_rng(0)
    indices = rng.integers(0, len(clusters), size=(1000, len(clusters)))
    draws = clusters['sum'].to_numpy()[indices].sum(axis=1) / clusters['count'].to_numpy()[indices].sum(axis=1)
    return values.kl.mean(), *np.quantile(draws, [.025, .975]), values.observed_fraction.mean()


def summarize_reference(table):
    values = table[(table.method == 'all_zero') & np.isclose(table.fraction, 1.)]
    if len(values) != 128 or values.window_id.nunique() != 128 or not np.isfinite(values.kl).all():
        raise ValueError('All-zero reference must have one finite value per test window')
    return values.kl.mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expanded', type=Path, default=ROOT / 'results/expanded/test')
    parser.add_argument('--depth', type=Path, default=ROOT / 'results/depth_extension/test')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/depth_extension/figures')
    args = parser.parse_args()
    figure = make_subplots(rows=1, cols=2, subplot_titles=['GPT-2', 'SmolLM2'])
    for column, (model, (total, layers)) in enumerate(MODEL_LAYERS.items(), 1):
        tables = {}
        for layer in layers:
            base = args.expanded if layer in ORIGINAL[model] else args.depth
            table = pd.read_csv(base / f'{model}_{layer}' / 'metrics.csv')
            if set(table.model) != {model} or set(table.layer) != {layer}:
                raise ValueError(f'Model or layer mismatch in {base}/{model}_{layer}')
            tables[layer] = table
        for method, color in COLORS.items():
            for random_control in (False, True):
                label = method + ('_random' if random_control else '')
                results = np.asarray([summarize(tables[layer], label) for layer in layers])
                name = method.replace('_centered', ' centered') + (' random' if random_control else '')
                custom = np.column_stack([layers, results[:, 3]])
                figure.add_trace(go.Scatter(x=np.asarray(layers) / (total - 1), y=results[:, 0], customdata=custom,
                    mode='lines+markers', name=name, legendgroup=name, showlegend=column == 1,
                    line=dict(color=color, dash='dot' if random_control else 'solid'),
                    error_y=dict(type='data', symmetric=False, array=np.maximum(0, results[:, 2] - results[:, 0]),
                                 arrayminus=np.maximum(0, results[:, 0] - results[:, 1]), visible=not random_control),
                    hovertemplate='Layer %{customdata[0]:.0f}<br>Relative depth %{x:.3f}<br>Mean KL %{y:.5f}'
                                  '<br>Mean observed removal %{customdata[1]:.3f}<extra>' + name + '</extra>'), row=1, col=column)
        references = np.asarray([summarize_reference(tables[layer]) for layer in layers])
        figure.add_trace(go.Scatter(x=np.asarray(layers) / (total - 1), y=references,
            mode='lines+markers', name='All zero reference', legendgroup='all_zero',
            showlegend=column == 1, visible='legendonly', line=dict(color='#333333', dash='dash'),
            customdata=np.asarray(layers),
            hovertemplate='Layer %{customdata:.0f}<br>Relative depth %{x:.3f}<br>Mean KL %{y:.5f}<extra>All zero reference</extra>'), row=1, col=column)
    figure.update_layout(template='plotly_white', title='Depth at 75% removal',
        height=700, legend_title_text='Selection (click to toggle)', margin=dict(l=75, r=180, t=100, b=165),
        updatemenus=[dict(type='buttons', direction='right', x=.74, y=1.15, showactive=True,
            buttons=[dict(label='Log scale', method='relayout', args=[{'yaxis.type': 'log', 'yaxis2.type': 'log'}]),
                     dict(label='Linear scale', method='relayout', args=[{'yaxis.type': 'linear', 'yaxis2.type': 'linear'}])])],
        annotations=list(figure.layout.annotations) + [dict(x=.5, y=-.22, xref='paper', yref='paper', showarrow=False,
            text='95% intervals: 1,000 article-cluster bootstrap replicates; four articles per model, so intervals are fragile.<br>'
                 'Independent y scales, logarithmic by default; centered rankings still zero activations.')])
    figure.update_xaxes(title_text='Relative depth: layer / (number of layers − 1)', range=[0, 1])
    figure.update_yaxes(title_text='Mean KL (nats per scored token)', type='log')
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / 'depth_comparison.html'
    figure.write_html(output, include_plotlyjs=True)
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
