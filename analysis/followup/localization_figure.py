#!/usr/bin/env python3
"""Plot position interventions and local projection geometry from saved results."""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/followup/localization'
METHODS = ['L', 'W', 'W_centered']
COLORS = {'L': '#D55E00', 'W': '#CC79A7', 'W_centered': '#009E73'}
NAMES = {'L': 'L: raw magnitude', 'W': 'W: weighted magnitude', 'W_centered': 'W: weighted distance from mean'}


def summarize(group, field):
    grouped = group.groupby('cluster_id')[field].agg(['sum', 'count'])
    if len(grouped) != 4 or len(group) != 128 or not np.isfinite(group[field]).all():
        raise ValueError('Expected 128 finite windows from four articles')
    rng = np.random.default_rng(1729)
    indexes = rng.integers(len(grouped), size=(1000, len(grouped)))
    draws = grouped['sum'].to_numpy()[indexes].sum(axis=1) / grouped['count'].to_numpy()[indexes].sum(axis=1)
    mean = group[field].mean()
    low, high = np.quantile(draws, [.025, .975])
    return mean, max(0., high-mean), max(0., mean-low)


metrics = pd.read_parquet(OUT / 'metrics.parquet')
geometry = pd.read_parquet(OUT / 'geometry.parquet')
fig = make_subplots(rows=2, cols=2, horizontal_spacing=.13, vertical_spacing=.18,
    subplot_titles=['Prefix-only ablation: positions 0–15', 'Suffix-only ablation: positions 16–127',
                    'Net removed vector norm on scored positions', 'Sum of individual norms on scored positions'])
for method in METHODS:
    for col, regime in enumerate(['prefix', 'suffix'], 1):
        group = metrics[(metrics.method == method) & (metrics.regime == regime)]
        mean, plus, minus = summarize(group, 'kl')
        fig.add_trace(go.Bar(x=[NAMES[method]], y=[mean], name=NAMES[method], legendgroup=method,
            showlegend=False, marker_color=COLORS[method],
            error_y=dict(type='data', symmetric=False, array=[plus], arrayminus=[minus]),
            hovertemplate=f'{NAMES[method]}<br>KL %{{y:.5f}} nats/scored token<extra></extra>'), row=1, col=col)
    for col, field in enumerate(['net_removed_norm', 'sum_individual_norms'], 1):
        group = geometry[(geometry.method == method) & (geometry.region == 'scored')]
        mean, plus, minus = summarize(group, field)
        fig.add_trace(go.Bar(x=[NAMES[method]], y=[mean], name=NAMES[method], legendgroup=method,
            showlegend=False, marker_color=COLORS[method],
            error_y=dict(type='data', symmetric=False, array=[plus], arrayminus=[minus]),
            hovertemplate=f'{NAMES[method]}<br>Mean norm %{{y:.3f}}<extra></extra>'), row=2, col=col)
fig.update_yaxes(title_text='KL (nats per scored token)', row=1, col=1)
fig.update_yaxes(title_text='KL (nats per scored token)', row=1, col=2)
fig.update_yaxes(title_text='Mean net vector norm', row=2, col=1)
fig.update_yaxes(title_text='Mean sum of individual norms', row=2, col=2)
fig.update_xaxes(tickangle=-10)
fig.update_layout(template='plotly_white', height=920, margin=dict(l=90, r=50, t=105, b=190),
    title='Where does the first-layer difference appear?',
    annotations=list(fig.layout.annotations) + [dict(x=0, y=-.19, xref='paper', yref='paper', xanchor='left', showarrow=False,
        align='left', text='Exploratory; 128 windows from four articles. Bars: means; intervals: 1,000 article-cluster bootstrap resamples.<br>'
        'Compare methods within a regime: prefix and suffix have different event budgets, and effects need not add.<br>'
        'Scored positions are 16–126. Net projection norm is descriptive, not causal mediation.')])
fig.write_html(OUT / 'localization.html', include_plotlyjs=True, config={'responsive': True})
print('Wrote', OUT / 'localization.html')
