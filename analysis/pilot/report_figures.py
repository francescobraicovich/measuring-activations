"""Generate publication figures from the saved calibration and test tables."""
from pathlib import Path
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'results/study'
DEST = STUDY / 'figures/report'
DEST.mkdir(parents=True, exist_ok=True)
METRICS = pd.read_parquet(STUDY / 'metrics.parquet')
MODELS = [('gpt2', 6, 'GPT-2 Small · layer 6'), ('smollm2', 16, 'SmolLM2-135M · layer 16')]
COLORS = {'N': '#0072B2', 'L': '#D55E00', 'G': '#009E73', 'W': '#B55A99', 'identity': '#777777', 'all_zero': '#202020'}
METHODS = ['N', 'L', 'G', 'W']
NOTES = 'Points: mean over windows; 95% intervals: article bootstrap (only four articles per model). Masks cover all 128 positions; KL scores positions 16–126.'


def interval(group):
    # Average random repetitions within each window before sampling clusters.
    window = group.groupby(['cluster_id', 'window_id'], as_index=False).agg(kl=('kl', 'mean'))
    clusters = window.groupby('cluster_id').kl.agg(['sum', 'count'])
    rng = np.random.default_rng(0)
    selection = rng.integers(0, len(clusters), size=(1000, len(clusters)))
    values = clusters['sum'].to_numpy()[selection].sum(axis=1) / clusters['count'].to_numpy()[selection].sum(axis=1)
    return np.quantile(values, [.025, .975])


def style(fig, title, footer=NOTES):
    fig.update_layout(title={'text': title, 'x': .04}, template='plotly_white', height=650, width=1200,
                      margin={'l': 65, 'r': 30, 't': 95, 'b': 180}, legend={'orientation': 'h', 'y': -0.14, 'x': 0},
                      font={'family': 'Arial, sans-serif', 'size': 13})
    fig.add_annotation(text=footer, xref='paper', yref='paper', x=0, y=-.29, xanchor='left', showarrow=False, font={'size': 10, 'color': '#555'})
    return fig


def causal(axis, axis_title, title):
    fig = make_subplots(rows=1, cols=2, subplot_titles=[m[2] for m in MODELS], shared_yaxes=False, horizontal_spacing=.12)
    for col, (model, layer, _) in enumerate(MODELS, 1):
        sub = METRICS[(METRICS.model == model) & (METRICS.layer == layer)]
        for label in METHODS + [m + '_random' for m in METHODS] + ['identity', 'all_zero']:
            table = sub[sub.method == label]
            if table.empty:
                continue
            points = []
            for fraction, group in table.groupby('fraction'):
                low, high = interval(group)
                points.append((fraction, group[axis].mean(), group.kl.mean(), low, high))
            points.sort()
            arr = np.array(points)
            base = label.replace('_random', '')
            random = label.endswith('_random')
            name = base + ' matched random' if random else {'identity': 'Identity', 'all_zero': 'All zero'}.get(label, label)
            fig.add_trace(go.Scatter(x=arr[:, 1], y=arr[:, 2], mode='markers' if label in ('identity', 'all_zero') else 'lines+markers',
                name=name, legendgroup=label, showlegend=col == 1,
                line={'color': COLORS[base], 'dash': 'dot' if random else 'solid'}, marker={'size': 7, 'symbol': 'circle-open' if random else 'circle'},
                error_y={'type': 'data', 'symmetric': False, 'array': np.maximum(0, arr[:, 4] - arr[:, 2]), 'arrayminus': np.maximum(0, arr[:, 2] - arr[:, 3]), 'visible': True},
                customdata=arr[:, [0, 3, 4]], hovertemplate='Nominal p=%{customdata[0]:.2f}<br>Observed x=%{x:.4f}<br>KL=%{y:.5f} nats/token<br>95% article interval=[%{customdata[1]:.5f}, %{customdata[2]:.5f}]<extra>%{fullData.name}</extra>'), row=1, col=col)
        fig.update_xaxes(title_text=axis_title, range=[-.025, 1.025], row=1, col=col)
        fig.update_yaxes(title_text='KL (nats per scored token)', rangemode='tozero', row=1, col=col)
    return style(fig, title)


def disagreement():
    fig = make_subplots(rows=1, cols=2, subplot_titles=[m[2] for m in MODELS], horizontal_spacing=.16)
    for col, (model, layer, _) in enumerate(MODELS, 1):
        df = pd.read_csv(STUDY / f'descriptive/{model}_{layer}/overlap.csv')
        df = df[np.isclose(df.fraction, .75)]
        z = np.zeros((4, 4))
        for r in df.itertuples():
            i, j = METHODS.index(r.left), METHODS.index(r.right)
            z[i, j] = z[j, i] = r.disagreement
        fig.add_trace(go.Heatmap(z=z, x=METHODS, y=METHODS, zmin=0, zmax=.4, coloraxis='coloraxis', text=z, texttemplate='%{text:.3f}', hovertemplate='%{y} vs %{x}<br>fraction of events disagreeing: %{z:.3f}<extra></extra>'), row=1, col=col)
        fig.update_yaxes(autorange='reversed', row=1, col=col)
    fig.update_layout(coloraxis={'colorscale': 'Blues', 'colorbar': {'title': 'Disagreement'}})
    return style(fig, 'Which events do the rankings disagree about?', 'Training activations only; nominal p = 0.75. Each cell is the fraction of token–channel events assigned different masks.')


def comparison():
    table = pd.read_csv(STUDY / 'paired_method_comparisons.csv')
    fig = make_subplots(rows=1, cols=2, subplot_titles=[m[2] for m in MODELS], horizontal_spacing=.12)
    for col, (model, layer, _) in enumerate(MODELS, 1):
        subset = table[(table.model == model) & (table.layer == layer)]
        for label, color in [('W minus L', '#8c4aa1'), ('N minus G', '#0072B2')]:
            part = subset[subset.comparison == label].sort_values('fraction')
            fig.add_trace(go.Scatter(x=part.fraction, y=part.kl_difference, mode='lines+markers', name=label, legendgroup=label, showlegend=col == 1,
                line={'color': color}, error_y={'type': 'data', 'symmetric': False, 'array': (part.article_high - part.kl_difference).clip(lower=0), 'arrayminus': (part.kl_difference - part.article_low).clip(lower=0)},
                customdata=np.stack([part.left_removal, part.right_removal, part.article_low, part.article_high], axis=1),
                hovertemplate='Nominal p=%{x:.2f}<br>KL difference=%{y:.5f}<br>Observed left=%{customdata[0]:.4f}; right=%{customdata[1]:.4f}<br>95% article interval=[%{customdata[2]:.5f}, %{customdata[3]:.5f}]<extra>%{fullData.name}</extra>'), row=1, col=col)
        fig.add_hline(y=0, line_dash='dash', line_color='#888', row=1, col=col)
        fig.update_xaxes(title_text='Nominal removal fraction p', row=1, col=col)
        fig.update_yaxes(title_text='Paired KL difference (nats/token)', row=1, col=col)
    return style(fig, 'Paired differences on the same windows', '95% intervals resample four articles. W/L remove exactly equal counts per token; N/G counts can differ.')


def depth():
    fig = make_subplots(rows=1, cols=2, subplot_titles=[m[2].split(' · ')[0] for m in MODELS], horizontal_spacing=.12)
    for col, (model, _, _) in enumerate(MODELS, 1):
        sub = METRICS[(METRICS.model == model) & (np.isclose(METRICS.fraction, .75) | (METRICS.method == 'all_zero'))]
        for label in METHODS + [m + '_random' for m in METHODS] + ['all_zero']:
            data = sub[sub.method == label]
            pts = []
            for order, depth_name in enumerate(['early', 'middle', 'final']):
                group = data[data.depth == depth_name]
                if len(group):
                    low, high = interval(group)
                    pts.append((order, depth_name, group.kl.mean(), low, high))
            if not pts:
                continue
            base = label.replace('_random', '')
            random = label.endswith('_random')
            fig.add_trace(go.Scatter(x=[p[1] for p in pts], y=[p[2] for p in pts], mode='lines+markers', name=(base + ' matched random' if random else 'All zero' if label == 'all_zero' else label), legendgroup=label, showlegend=col == 1,
                line={'color': COLORS[base], 'dash': 'dot' if random else 'solid'}, marker={'symbol': 'circle-open' if random else 'circle'},
                error_y={'type': 'data', 'symmetric': False, 'array': [max(0,p[4]-p[2]) for p in pts], 'arrayminus': [max(0,p[2]-p[3]) for p in pts]}), row=1, col=col)
        fig.update_yaxes(title_text='KL (nats per scored token)', rangemode='tozero', row=1, col=col)
    return style(fig, 'Depth check at p = 0.75', NOTES + ' All-zero is the layer reference; three depths do not imply a universal trend.')


figures = {
    'causal_events': causal('observed_fraction', 'Mean observed fraction of events removed', 'What happens when we remove weak activations?'),
    'causal_absolute_mass': causal('removed_abs_fraction', 'Mean per-window fraction of |activation| removed', 'KL versus removed activation magnitude'),
    'causal_weighted_mass': causal('removed_weighted_fraction', 'Mean per-window fraction of weighted magnitude removed', 'KL versus removed output-weighted magnitude'),
    'mask_disagreement': disagreement(),
    'paired_methods': comparison(),
    'depth': depth(),
}
os.environ.setdefault('BROWSER_PATH', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
for name, fig in figures.items():
    fig.write_html(DEST / f'{name}.html', include_plotlyjs=True)
    for suffix in ('svg', 'png'):
        try:
            fig.write_image(DEST / f'{name}.{suffix}', scale=2 if suffix == 'png' else 1)
        except Exception as error:
            print(f'{name}.{suffix}: {error}', flush=True)
    print(f'Wrote {name}', flush=True)
