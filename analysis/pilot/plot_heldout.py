"""Plot held-out, per-token removal quotas from saved diagnostics."""
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2] / 'results/study'
fig = make_subplots(rows=1, cols=2, subplot_titles=['GPT-2, layer 6', 'SmolLM2, layer 16'])
summary = []
for column, label in enumerate(('gpt2_6', 'smollm2_16'), 1):
    table = pd.read_parquet(ROOT / 'heldout_diagnostics' / label / 'token_removal.parquet')
    table = table[table.fraction == .75].copy()
    table['removed_fraction'] = table.removed_count / table.channel_count
    for method, color in (('N', '#0072B2'), ('G', '#009E73')):
        values = table.loc[table.method == method, 'removed_fraction']
        if len(values) != 128 * 128:
            raise ValueError(f'Unexpected token count for {label}, {method}')
        summary.append({'model_layer': label, 'method': method, 'min': values.min(),
                        'p05': values.quantile(.05), 'median': values.median(),
                        'p95': values.quantile(.95), 'max': values.max(), 'mean': values.mean()})
        fig.add_trace(go.Histogram(x=values, name=method, legendgroup=method,
                      showlegend=column == 1, marker_color=color, opacity=.55,
                      histnorm='probability', xbins=dict(start=0, end=1, size=.02),
                      hovertemplate='Removed fraction: %{x:.2f}<br>Share of tokens: %{y:.1%}<extra>'+method+'</extra>'), row=1, col=column)
    fig.add_vline(x=.75, line_dash='dash', line_color='#333333', row=1, col=column)
fig.update_layout(title='Held-out p=.75: N/G removal varies by token', barmode='overlay',
                  template='plotly_white', legend_title_text='Ranking',
                  annotations=list(fig.layout.annotations)+[dict(text='Dashed line: L and W each remove exactly 75% per token',
                  x=.5, y=-.25, xref='paper', yref='paper', showarrow=False)])
fig.update_xaxes(title_text='Fraction of channels removed', range=[0, 1])
fig.update_yaxes(title_text='Share of tokens per 0.02 bin', tickformat='.0%', row=1, col=1)
fig.update_yaxes(tickformat='.0%', row=1, col=2)
path = ROOT / 'figures' / 'causal' / 'heldout_token_removal.html'
fig.write_html(path, include_plotlyjs=True)
fig.write_image(path.with_suffix('.svg'))
fig.write_image(path.with_suffix('.png'), scale=2)
pd.DataFrame(summary).to_csv(ROOT / 'heldout_diagnostics' / 'token_removal_summary.csv', index=False)
print(pd.DataFrame(summary).to_string(index=False))
print('Wrote', path)
