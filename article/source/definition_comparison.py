"""Build definition comparisons from the saved individual- and all-layer summaries."""
import json

import pandas as pd

METHODS = ('N', 'L', 'G', 'W')
FRACTIONS = (.1, .2, .3, .4, .5, .6, .7, .75, .8, .85, .9, .95)
LAYERS = {'gpt2': (0, 2, 6, 8, 11, -1), 'smollm2': (0, 5, 16, 21, 29, -1)}
LABELS = {'N': 'Neuron percentile', 'L': 'Within-token magnitude',
          'G': 'Pooled percentile', 'W': 'Weight adjusted'}


def load_summary(results):
    return pd.concat([pd.read_csv(results / 'data/summary.csv')] + [
        pd.read_csv(results / 'all_layers_combined' / model / 'summary.csv')
        for model in LAYERS
    ], ignore_index=True)


METRICS = {'kl': 'KL divergence', 'flip_rate': 'Token flip rate'}


def comparison_data(summary):
    selected = summary[summary.method.isin(METHODS) & summary.metric.isin([*METRICS, 'observed_fraction'])]
    data = {}
    for model, layers in LAYERS.items():
        data[model] = {}
        for layer in layers:
            data[model][str(layer)] = {}
            for method in METHODS:
                values = selected[(selected.model == model) & (selected.layer == layer) & (selected.method == method)]
                points = values.pivot(index='fraction', columns='metric', values='mean').sort_index()
                columns = ['observed_fraction', *METRICS]
                if len(points) != 16 or not set(FRACTIONS).issubset(points.index) or points[columns].isna().any().any():
                    raise ValueError(f'Missing comparison data for {model} layer {layer} {method}')
                data[model][str(layer)][method] = points[columns].reset_index().to_numpy().tolist()
    return data


def comparison_html(results, colors):
    payload = {'models': comparison_data(load_summary(results)), 'layers': LAYERS,
               'fractions': FRACTIONS, 'metrics': list(METRICS), 'definitions': LABELS,
               'colors': {method: colors[method] for method in METHODS}}
    encoded = json.dumps(payload, separators=(',', ':'), allow_nan=False)
    metrics = ''.join(f'<option value="{key}">{label}</option>' for key, label in METRICS.items())
    options = ''.join(f'<option value="{method}">{method} · {LABELS[method]}</option>' for method in METHODS)
    controls = f'<div class="comparison-controls"><label>Metric <select id="winner-metric">{metrics}</select></label></div>'
    winners = '<div id="winner-matrices"></div>'
    comparison = f'''<div class="comparison-controls pairwise-controls"><label>Metric <select id="comparison-metric">{metrics}</select></label><label>First definition <select id="comparison-first">{options}</select></label><label>Second definition <select id="comparison-second">{options}</select></label></div>
    <p class="comparison-direction" id="comparison-direction" aria-live="polite"></p>
    <div id="comparison-gradient" class="comparison-gradient" aria-hidden="true"></div><div class="comparison-gradient-labels"><span id="comparison-better-first"></span><span>1× · Equal</span><span id="comparison-better-second"></span></div>
    <div id="comparison-matrices"></div>
    <noscript><p>Enable JavaScript to view the definition matrices.</p></noscript>
    <script type="application/json" id="comparison-data">{encoded}</script>'''
    return controls, winners, comparison
