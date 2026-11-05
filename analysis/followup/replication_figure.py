#!/usr/bin/env python3
"""Create an interactive paired-article plot from the saved replication."""
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'results/followup/replication'
frame = pd.read_parquet(DIRECTORY / 'metrics.parquet')
paired = frame[frame.method.isin(['L', 'W', 'W_centered'])].pivot(index='cluster_id', columns='method', values='kl')
if len(paired) != 32 or paired.isna().any().any():
    raise ValueError('Expected three results on each of 32 articles')
figure = make_subplots(rows=1, cols=2, subplot_titles=['W compared with L', 'Centered W compared with W'])
for col, (x, y, color) in enumerate([('L', 'W', '#ad397e'), ('W', 'W_centered', '#00798a')], 1):
    maximum = max(paired[x].max(), paired[y].max()) * 1.06
    figure.add_trace(go.Scatter(x=[0, maximum], y=[0, maximum], mode='lines', name='Equal KL',
                               line={'color': '#999', 'dash': 'dash'}, showlegend=col == 1,
                               hoverinfo='skip'), row=1, col=col)
    figure.add_trace(go.Scatter(x=paired[x], y=paired[y], mode='markers', name='New test article',
                               customdata=paired.index, marker={'color': color, 'size': 8, 'opacity': .75},
                               showlegend=col == 1,
                               hovertemplate='Article %{customdata}<br>' + x + ' KL %{x:.4f}<br>' + y + ' KL %{y:.4f}<extra></extra>'), row=1, col=col)
    figure.add_trace(go.Scatter(x=[paired[x].mean()], y=[paired[y].mean()], mode='markers', name='Mean across articles',
                               marker={'symbol': 'diamond', 'size': 13, 'color': '#222', 'line': {'color': 'white', 'width': 1}},
                               showlegend=col == 1,
                               hovertemplate='Article mean<br>' + x + ' KL %{x:.4f}<br>' + y + ' KL %{y:.4f}<extra></extra>'), row=1, col=col)
    figure.update_xaxes(title_text=x.replace('_centered', ' centered') + ' KL', range=[0, maximum], row=1, col=col)
    figure.update_yaxes(title_text=y.replace('_centered', ' centered') + ' KL', range=[0, maximum], row=1, col=col)
figure.update_layout(template='plotly_white', title='Does the first-layer result extend to new articles?', height=650,
    margin={'l': 75, 'r': 40, 't': 100, 'b': 155}, legend={'orientation': 'h', 'y': -.20},
    annotations=list(figure.layout.annotations) + [dict(x=.5, y=-.34, xref='paper', yref='paper', showarrow=False,
        text='Targeted exploratory replication; 32 new articles, one window each. Both masks remove 75% per token.<br>'
             'KL is in nats per scored token; selection was made after inspecting the original four articles.')])
figure.write_html(DIRECTORY / 'paired_articles.html', include_plotlyjs=True, config={'responsive': True})
print(DIRECTORY / 'paired_articles.html')
