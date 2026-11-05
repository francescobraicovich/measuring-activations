#!/usr/bin/env python3
"""Generate dense removed-mass figures from saved per-window metrics."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.expanded.figures import COLORS, ROOT, TARGETS, interval


def create_figure(tables, axis, axis_title, title):
    figure = make_subplots(rows=1, cols=2, subplot_titles=[target[1] for target in TARGETS],
                           shared_yaxes=False, horizontal_spacing=.12)
    for column, table in enumerate(tables, 1):
        if table.cluster_id.nunique() != 4:
            raise ValueError('Expected four held-out articles per model')
        for method in list(COLORS) + [name + '_random' for name in COLORS] + ['identity', 'all_zero']:
            selected = table[table.method == method]
            if selected.empty:
                raise ValueError(f'Missing method: {method}')
            points = []
            for fraction, group in selected.groupby('fraction'):
                if group.duplicated(['window_id', 'repetition']).any():
                    raise ValueError(f'Duplicate observations: {method}, {fraction}')
                if not group.groupby('window_id').cluster_id.nunique().eq(1).all():
                    raise ValueError('Inconsistent article identifiers')
                if not np.isfinite(group[['kl', axis]].to_numpy()).all():
                    raise ValueError(f'Non-finite observations: {method}, {fraction}')
                windows = group.groupby(['window_id', 'cluster_id'], as_index=False)[['kl', axis]].mean()
                mean, low, high = interval(windows)
                points.append((fraction, windows[axis].mean(), mean, low, high))
            values = np.asarray(points)
            random = method.endswith('_random')
            reference = method in ('identity', 'all_zero')
            base = method.removesuffix('_random')
            name = base + ' matched random' if random else {'identity': 'Identity', 'all_zero': 'All zero'}.get(method, method)
            color = COLORS.get(base, '#777777' if method == 'identity' else '#202020')
            figure.add_trace(go.Scatter(
                x=values[:, 1], y=values[:, 2], mode='markers' if reference else 'lines+markers',
                name=name, legendgroup=method, showlegend=column == 1,
                line=dict(color=color, dash='dot' if random else 'solid'),
                marker=dict(size=7, symbol='circle-open' if random else 'circle'),
                error_y=dict(type='data', symmetric=False, array=np.maximum(0, values[:, 4] - values[:, 2]),
                             arrayminus=np.maximum(0, values[:, 2] - values[:, 3])),
                customdata=values[:, [0, 3, 4]],
                hovertemplate='Nominal p=%{customdata[0]:.2f}<br>Mean removed fraction=%{x:.4f}'
                              '<br>Mean KL=%{y:.5f} nats/token<br>95% article interval='
                              '[%{customdata[1]:.5f}, %{customdata[2]:.5f}]<extra>%{fullData.name}</extra>'),
                row=1, col=column)
        figure.update_xaxes(title_text=axis_title, range=[-.025, 1.025], row=1, col=column)
        figure.update_yaxes(title_text='Mean KL (nats per scored token)', rangemode='tozero', row=1, col=column)
    figure.update_layout(title=dict(text=title, x=.04), template='plotly_white', height=720, autosize=True,
                         margin=dict(l=65, r=30, t=95, b=250), legend=dict(orientation='h', y=-.16, x=0),
                         font=dict(family='Arial, sans-serif', size=13))
    figure.add_annotation(text='Means over windows; 95% intervals: 1,000 article-bootstrap replicates '
                               '(four articles per model; intervals are fragile).<br>'
                               'Independent y scales. Masks cover all 128 positions; KL scores positions 16–126.',
                          xref='paper', yref='paper', x=0, y=-.47, xanchor='left', showarrow=False,
                          align='left', font=dict(size=10, color='#555'))
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'results/expanded/test')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/expanded/figures')
    args = parser.parse_args()
    tables = [pd.read_csv(args.input / label / 'metrics.csv') for label, _ in TARGETS]
    args.output.mkdir(parents=True, exist_ok=True)
    for name, axis, axis_title, title in (
        ('causal_absolute_mass', 'removed_abs_fraction', 'Mean per-window fraction of |activation| removed',
         'KL versus removed activation magnitude'),
        ('causal_weighted_mass', 'removed_weighted_fraction', 'Mean per-window fraction of weighted magnitude removed',
         'KL versus removed output-weighted magnitude'),
    ):
        path = args.output / f'{name}.html'
        create_figure(tables, axis, axis_title, title).write_html(path, include_plotlyjs=True, config={'responsive': True})
        print(f'Wrote {path}')


if __name__ == '__main__':
    main()
