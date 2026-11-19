"""Draw middle-layer training channel scale against outgoing weight norms."""
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

root = Path(__file__).resolve().parents[2]
figure = make_subplots(rows=1, cols=2, subplot_titles=['GPT-2 · layer 6', 'SmolLM2 · layer 16'], horizontal_spacing=.13)
for col, label in enumerate(['gpt2_6', 'smollm2_16'], 1):
    data = pd.read_csv(root / 'results/study/descriptive' / label / 'channels.csv')
    valid = data[(data.rms > 0) & (data.weight_norm > 0)].copy()
    rho = spearmanr(data.rms, data.weight_norm).statistic
    figure.add_trace(go.Scatter(x=valid.rms, y=valid.weight_norm, mode='markers',
        customdata=valid[['channel','positive_fraction','weighted_rms']],
        marker=dict(size=4, opacity=.65, color=valid.positive_fraction, coloraxis='coloraxis'),
        hovertemplate='channel %{customdata[0]}<br>RMS %{x:.4g}<br>weight norm %{y:.4g}<br>positive fraction %{customdata[1]:.3f}<br>weighted RMS %{customdata[2]:.4g}<extra></extra>',
        showlegend=False), row=1, col=col)
    valid['bin'] = pd.qcut(valid.rms, 10, duplicates='drop')
    trend = valid.groupby('bin', observed=True)[['rms', 'weight_norm']].median()
    figure.add_trace(go.Scatter(x=trend.rms, y=trend.weight_norm, mode='lines+markers',
        line=dict(color='#262626', width=2), marker=dict(size=5), name='Binned median', showlegend=col == 1), row=1, col=col)
    figure.update_xaxes(type='log', title_text='Channel activation RMS', row=1, col=col)
    figure.update_yaxes(type='log', title_text='Outgoing weight norm', row=1, col=col)
    figure.add_annotation(x=.02 if col == 1 else .59, y=.96, xref='paper', yref='paper', text=f'Spearman ρ = {rho:.3f}', showarrow=False, xanchor='left')
figure.update_layout(title='Activation scale and outgoing weights · calibration', template='plotly_white',
    coloraxis=dict(colorscale='Viridis', cmin=0, cmax=1, colorbar=dict(title='Positive<br>fraction', x=1.03)),
    height=590, margin=dict(l=75,r=90,t=100,b=80), legend=dict(orientation='h', y=-.19))
out=root/'results/study/figures/report/scale_weight'
figure.write_html(str(out)+'.html', include_plotlyjs=True, config={'responsive':True})
figure.write_image(str(out)+'.svg', width=1200, height=590)
figure.write_image(str(out)+'.png', width=1200, height=590, scale=2)
