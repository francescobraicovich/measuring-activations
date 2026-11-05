#!/usr/bin/env python3
"""Build the article, checked numerical claims, figures, and embedded explorer."""
import base64
import json
import math
from pathlib import Path
import re
import subprocess

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = ROOT / 'results/large_study'
MATH_ASSETS = ROOT / 'assets/vendor/katex'


def measurement(table, **keys):
    selected = table
    for column, value in keys.items():
        selected = selected[selected[column] == value]
    if len(selected) != 1:
        raise ValueError(f'Expected one measurement for {keys}; found {len(selected)}')
    return selected.iloc[0]


def format_uncertainty(mean, lower, upper):
    """Round the mean to the same decimal place as the approximate uncertainty."""
    uncertainty = (upper - lower) / 2
    if uncertainty == 0:
        return f'{mean:+.3g} ± 0'
    decimals = 1 - math.floor(math.log10(uncertainty))
    rounded_uncertainty = round(uncertainty, decimals)
    if decimals >= 0:
        return f'{mean:+.{decimals}f} ± {rounded_uncertainty:.{decimals}f}'
    return f'{round(mean, decimals):+.0f} ± {rounded_uncertainty:.0f}'


def article_values(summary, paired):
    """Resolve interpolated empirical claims from saved measurements."""
    values = {}
    cases = {
        'smol_l': ('smollm2', 16, 'L'), 'smol_g': ('smollm2', 16, 'G'), 'smol_w': ('smollm2', 16, 'W'),
        'smol_random': ('smollm2', 16, 'random'),
        'gpt_l': ('gpt2', 6, 'L'), 'gpt_n': ('gpt2', 6, 'N'), 'gpt_random': ('gpt2', 6, 'random'),
        'first_l': ('gpt2', 0, 'L'), 'first_w': ('gpt2', 0, 'W'),
        'first_centered': ('gpt2', 0, 'W_centered'),
        'last_n': ('smollm2', 29, 'N'), 'last_g': ('smollm2', 29, 'G'),
    }
    for name, (model, layer, method) in cases.items():
        for metric in ('kl', 'observed_fraction', 'removed_weighted_fraction', 'flip_rate'):
            row = measurement(summary, model=model, layer=layer, method=method, fraction=.75, metric=metric)
            values[f'{name}_{metric}'] = f'{row["mean"]:.3g}' if metric == 'kl' else f'{100 * row["mean"]:.3g}'
    for model, winner in [('smollm2', 'G'), ('gpt2', 'N')]:
        for metric in ('kl', 'flip_rate'):
            subset = summary[(summary.model == model) & (summary.fraction == .75) &
                             (summary.metric == metric) & summary.method.isin(['N', 'L', 'G', 'W'])]
            if len(subset) != 20 or any(group.loc[group['mean'].idxmin(), 'method'] != winner
                                         for _, group in subset.groupby('layer')):
                raise ValueError(f'The stated {model} {metric} ranking no longer holds')
    for fraction, label in [(0.5, '50'), (0.9, '90'), (0.99, '99')]:
        for metric in ('kl', 'flip_rate', 'observed_fraction'):
            row = measurement(summary, model='smollm2', layer=16, method='G', fraction=fraction, metric=metric)
            values[f'smol_g_{label}_{metric}'] = f'{row["mean"]:.3g}' if metric == 'kl' else f'{100 * row["mean"]:.3g}'
    for fraction, label in [(0.9, '90'), (0.99, '99')]:
        row = measurement(summary, model='gpt2', layer=6, method='L', fraction=fraction, metric='kl')
        values[f'gpt_l_{label}_kl'] = f'{row["mean"]:.3g}'
    for name, model, layer, method, reference in [
        ('smol_difference', 'smollm2', 16, 'L', 'random'),
        ('gpt_difference', 'gpt2', 6, 'L', 'random'),
        ('first_difference', 'gpt2', 0, 'W', 'L'),
        ('smol_weight_difference', 'smollm2', 16, 'W', 'L'),
    ]:
        row = measurement(paired, model=model, layer=layer, method=method, reference=reference, fraction=.75, metric='kl')
        values[name] = format_uncertainty(row["mean"], row.lower, row.upper)
    for metric in ('observed_fraction', 'removed_abs_fraction'):
        row = measurement(summary, model='smollm2', layer=16, method='L', fraction=.99, metric=metric)
        values[f'p99_{metric}'] = f'{100 * row["mean"]:.1f}'
    geometry = pd.read_parquet(ROOT / 'results/followup/localization/geometry.parquet')
    for method in ('L', 'W'):
        group = geometry[(geometry.method == method) & (geometry.region == 'scored')]
        if len(group) != 128 or group.cluster_id.nunique() != 4:
            raise ValueError('Unexpected geometry sample')
        for field in ('net_removed_norm', 'sum_individual_norms'):
            values[f'geometry_{method}_{field}'] = f'{group[field].mean():.3g}'
    for model, layer, method, reference, direction in [
        ('smollm2', 16, 'L', 'random', -1), ('gpt2', 6, 'L', 'random', 1),
        ('gpt2', 0, 'W', 'L', 1),
    ]:
        row = measurement(paired, model=model, layer=layer, method=method, reference=reference, fraction=.75, metric='kl')
        if not (row['mean'] * direction > 0 and (row.lower if direction > 0 else -row.upper) > 0):
            raise ValueError('A stated paired comparison no longer holds')
    return values


def substitute(template, values):
    def replace(match):
        name = match.group(1)
        if name not in values:
            raise ValueError(f'Unresolved article value: {name}')
        return str(values[name])
    return re.sub(r'\[\[([a-zA-Z0-9_]+)\]\]', replace, template)


def inline_svg(path, title, description):
    source = path.read_text()
    source = source[source.index('<svg'):]
    identifier = path.stem.replace('_', '-')
    # Matplotlib reuses names such as figure_1 and clip-path identifiers.
    # Namespace them so separate inline figures cannot clip one another.
    for old in re.findall(r'\bid="([^"]+)"', source):
        new = f'{identifier}-{old}'
        source = source.replace(f'id="{old}"', f'id="{new}"')
        source = source.replace(f'url(#{old})', f'url(#{new})')
        source = source.replace(f'href="#{old}"', f'href="#{new}"')
    source = source.replace('<svg ', f'<svg role="img" aria-labelledby="{identifier}-title {identifier}-description" ', 1)
    end = source.index('>') + 1
    return (source[:end] + f'<title id="{identifier}-title">{title}</title>'
            f'<desc id="{identifier}-description">{description}</desc>' + source[end:])


def typeset_math(source):
    return subprocess.run(['node', str(HERE / 'render_math.js')], input=source,
                          text=True, capture_output=True, check=True).stdout


def math_styles():
    """Embed WOFF2 fonts so the standalone article has no network dependency."""
    styles = (MATH_ASSETS / 'katex.min.css').read_text()
    def embed_font(match):
        encoded = base64.b64encode((MATH_ASSETS / match.group(1)).read_bytes()).decode('ascii')
        return f'url(data:font/woff2;base64,{encoded}) format("woff2")'
    # The official stylesheet lists legacy WOFF/TTF fallbacks. WOFF2 suffices
    # for supported browsers and keeps this single-file report smaller.
    return re.sub(r'url\((fonts/[^)]+\.woff2)\) format\("woff2"\)(?:,url\([^)]+\) format\("(?:woff|truetype)"\))*', embed_font, styles)


def main():
    from report_figures import build_figures
    build_figures()
    summary = pd.read_csv(RESULTS / 'data/summary.csv')
    paired = pd.read_csv(RESULTS / 'data/paired_comparisons.csv')
    values = article_values(summary, paired)
    values['definitions_html'] = (HERE / 'strength_definitions.html').read_text()
    values['report_styles'] = (HERE / 'report.css').read_text()
    values['comparison_figure'] = inline_svg(
        RESULTS / 'figures/ablation_comparison.svg',
        'Removing the same fraction can have very different effects',
        'Logarithmic KL axes show L, W, and random removal at 75 percent in all five sampled layers of each model. '
        'L is below random in every sampled SmolLM2 layer, but above random at GPT-2 layer 6. W has higher KL than L at GPT-2 layer 0. '
        'Whiskers show 95 percent article bootstrap intervals; values are available in the explorer.')
    values['geometry_figure'] = inline_svg(
        RESULTS / 'figures/projection_geometry.svg',
        'A smaller upper bound can accompany a larger net perturbation',
        'GPT-2 layer 0, 75 percent removal, 128 windows from four articles. '
        'The sum of individual removed contribution norms falls from 481 under L to 465 under W, '
        'while the net removed vector norm rises from 12.1 to 16.6. '
        'The two panels have different scales; units are internal projection coordinates.')
    source = (RESULTS / 'interactive.html').read_text()
    introduction = '<h1>Activation ablation study</h1><p>Explore held-out measurements by model, layer, method and nominal removal fraction.</p>'
    if introduction not in source or '</main><script>' not in source:
        raise ValueError('Explorer template changed')
    explorer_split = '<h2>How the selected outcome varies across layers</h2>'
    if source.count(introduction) != 1 or source.count(explorer_split) != 1:
        raise ValueError('Explorer sections changed')
    before, after = source.split(introduction, 1)
    main_plot, remaining = after.split(explorer_split, 1)
    values['results_explorer'] = ('<section id="data-explorer" class="explorer"><h2>Results</h2>'
                                  + main_plot + '</section>')
    article = typeset_math(substitute((HERE / 'report_content.html').read_text(), values))
    source = before + article + '<section class="explorer"><h2>More results</h2>' + explorer_split + remaining
    source = source.replace('</main><script>', '</section></main><script>', 1)
    source = source.replace('<title>Activation ablation study</title>', '<title>What is a weak activation? Four definitions compared</title>', 1)
    source = source.replace('</style></head>', math_styles() + values['report_styles'] + '</style></head>', 1)
    source = source.replace('</head>', '<script>' + (MATH_ASSETS / 'katex.min.js').read_text() + '</script></head>', 1)
    source = source.replace('</body>', '<script>' + (HERE / 'strength_definitions.js').read_text() + '</script></body>', 1)
    (ROOT / 'blog.html').write_text(source)
    numeric_values = {key: value for key, value in values.items() if not key.endswith(('_html', '_styles', '_figure', '_explorer'))}
    (RESULTS / 'article_values.json').write_text(json.dumps(numeric_values, indent=2) + '\n')
    print('Wrote blog.html with checked claims and self-contained figures')


if __name__ == '__main__':
    main()
