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
    for metric in ('kl', 'flip_rate'):
        partial = measurement(summary, model='gpt2', layer=6, method='G', fraction=.9, metric=metric)['mean']
        full = measurement(summary, model='gpt2', layer=6, method='G', fraction=1., metric=metric)['mean']
        random = measurement(summary, model='gpt2', layer=6, method='random', fraction=.9, metric=metric)['mean']
        if partial <= max(full, random):
            raise ValueError(f'The stated GPT-2 layer 6 comparison no longer holds for {metric}')
        for label, mean in [('90', partial), ('100', full)]:
            values[f'gpt_g_{label}_{metric}'] = f'{mean:.3g}' if metric == 'kl' else f'{100 * mean:.3g}'
    for metric in ('kl', 'flip_rate'):
        subset = summary[(summary.model == 'smollm2') & (summary.metric == metric) &
                         summary.method.isin(['N', 'L', 'G', 'W', 'random'])]
        baseline = subset[subset.method == 'random'][['layer', 'fraction', 'mean']]
        for method in ('N', 'L', 'G', 'W'):
            rows = subset[subset.method == method]
            if len(rows) != 80 or any((group.sort_values('fraction')['mean'].diff().dropna() < -1e-10).any()
                                      for _, group in rows.groupby('layer')):
                raise ValueError(f'The stated SmolLM2 monotonicity no longer holds for {method} {metric}')
            compared = rows.merge(baseline, on=['layer', 'fraction'], suffixes=('_method', '_random'))
            interior = compared[(compared.fraction > 0) & (compared.fraction < 1)]
            if len(interior) != 75 or (interior.mean_method >= interior.mean_random).any():
                raise ValueError(f'The stated SmolLM2 random comparison no longer holds for {method} {metric}')
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
    for model, layer, method, reference, direction in [
        ('smollm2', 16, 'L', 'random', -1), ('gpt2', 6, 'L', 'random', 1),
        ('gpt2', 0, 'W', 'L', 1),
    ]:
        row = measurement(paired, model=model, layer=layer, method=method, reference=reference, fraction=.75, metric='kl')
        if not (row['mean'] * direction > 0 and (row.lower if direction > 0 else -row.upper) > 0):
            raise ValueError('A stated paired comparison no longer holds')
    return values



def window_values():
    from figures import window_flip_rates

    rates = window_flip_rates()
    return {'smol_window_median': f'{100 * rates.quantile(.5):.2g}',
            'smol_window_p90': f'{100 * rates.quantile(.9):.2g}',
            'smol_window_max': f'{100 * rates.max():.2g}'}


def sequential_values():
    """Read all-layer measurements used in the article from completed runs."""
    values = {}
    for model, prefix, method in [('gpt2', 'all_gpt', 'N'), ('smollm2', 'all_smol', 'G')]:
        directory = RESULTS / 'all_layers_combined' / model
        run = json.loads((directory / 'run.json').read_text())
        if run['status'] != 'passed' or run['windows'] != 64 or run['articles'] != 61:
            raise ValueError(f'Unexpected all-layer run: {model}')
        summary = pd.read_csv(directory / 'summary.csv')
        for name, ranking in [(prefix, method), (prefix + '_local', 'L'), (prefix + '_random', 'random')]:
            for metric in ('kl', 'flip_rate', 'observed_fraction'):
                row = measurement(summary, model=model, layer=-1, method=ranking, fraction=.75, metric=metric)
                values[f'{name}_{metric}'] = f'{row["mean"]:.3g}' if metric == 'kl' else f'{100 * row["mean"]:.3g}'
    return values


def substitute(template, values):
    def replace(match):
        name = match.group(1)
        if name not in values:
            raise ValueError(f'Unresolved article value: {name}')
        return str(values[name])
    return re.sub(r'\[\[([a-zA-Z0-9_]+)\]\]', replace, template)


def render_sidenotes(source):
    """Use the same linked footnotes and progressive enhancement as the blog."""
    footnotes = []
    for number in (1, 2, 3):
        pattern = (rf'<span class="sidenote"><button class="sidenote-trigger"[^>]*>{number}</button>'
                   rf'<span class="sidenote-content" id="sidenote-{number}" role="note">(.*?)</span></span>')
        match = re.search(pattern, source, re.S)
        if not match:
            raise ValueError(f'Footnote {number} was not found')
        footnotes.append(f'<li id="fn{number}" class="footnote-item"><p>{match.group(1)} <a href="#fnref{number}" class="footnote-backref">↩︎</a></p></li>')
        reference = f'<sup class="footnote-ref"><a href="#fn{number}" id="fnref{number}">[{number}]</a></sup>'
        source = source[:match.start()] + reference + source[match.end():]
    notes = '<hr class="footnotes-sep"><section class="footnotes"><ol class="footnotes-list">' + ''.join(footnotes) + '</ol></section>'
    return source.replace('<article class="article">', '<article class="article prose">', 1).replace('</article>', notes + '</article>', 1)


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
    from figures import COLORS, build_figures, window_distributions
    from definition_comparison import comparison_html
    from explorer import build as build_explorer

    build_explorer(RESULTS / 'data', RESULTS / 'interactive.html')
    build_figures()
    summary = pd.read_csv(RESULTS / 'data/summary.csv')
    paired = pd.read_csv(RESULTS / 'data/paired_comparisons.csv')
    values = article_values(summary, paired)
    values.update(sequential_values())
    values.update(window_values())
    values['definitions_html'] = (HERE / 'strength_definitions.html').read_text()
    values['report_styles'] = (HERE / 'article.css').read_text()
    values['window_distribution_html'] = json.dumps(window_distributions(), separators=(',', ':'), allow_nan=False)
    values['window_flip_rate_figure'] = inline_svg(
        RESULTS / 'figures/window_flip_rates.svg',
        'Distribution of next-token flip rates across windows',
        'Histogram of 512 held-out windows for definition G at 75 percent nominal removal in SmolLM2 layer 16. '
        'Vertical lines mark the median, 90th percentile, and maximum.')
    values['comparison_figure'] = inline_svg(
        RESULTS / 'figures/ablation_comparison.svg',
        'Middle-layer token flip rates under weak and random removal',
        'Top-token flip rates versus mean observed removal in GPT-2 layer 6 and SmolLM2 layer 16. '
        'N, G, and random removal are shown with 95 percent article-bootstrap intervals. '
        'In GPT-2, G rises above random before returning to the common all-zero endpoint.')
    values['comparison_metric_html'], values['definition_winners_html'], values['definition_comparison_html'] = comparison_html(RESULTS, COLORS)
    source = (RESULTS / 'interactive.html').read_text()
    introduction = '<h1>Activation ablation study</h1><p>Explore held-out measurements by model, layer, method and nominal removal fraction.</p>'
    if introduction not in source or '</main><script>' not in source:
        raise ValueError('Explorer template changed')
    explorer_split = '<div class="controls" id="depth-controls">'
    if source.count(introduction) != 1 or source.count(explorer_split) != 1:
        raise ValueError('Explorer sections changed')
    before, after = source.split(introduction, 1)
    main_plot, remaining = after.split(explorer_split, 1)
    values['results_explorer'] = ('<section id="data-explorer" class="explorer"><h2>Experiments &amp; Results</h2>'
                                  + '<p>I zero selected activations just before the MLP output projection. '
                                  + 'The single-layer runs use 512 windows from 61 WikiText-2 articles.</p>'
                                  + '<h3 class="plot-title">Model output changes as more activations are removed</h3>' + main_plot + '</section>')
    article = typeset_math(render_sidenotes(substitute((HERE / 'article.html').read_text(), values)))
    source = before + article + '<details class="explorer additional-results"><summary>Additional Results</summary>' + explorer_split + remaining
    source = source.replace('</main><script>', '</details></main><script>', 1)
    source = source.replace('<title>Activation ablation study</title>', '<title>What is a weak activation? Four definitions compared</title>', 1)
    source = source.replace('</style></head>', math_styles() + values['report_styles'] + '</style></head>', 1)
    source = source.replace('</head>', '<script>' + (MATH_ASSETS / 'katex.min.js').read_text() + '</script></head>', 1)
    source = source.replace('</body>', '<script>' + (HERE / 'strength_definitions.js').read_text() + '</script><script>' + (HERE / 'footnotes.js').read_text() + '</script><script>' + (HERE / 'definition_comparison.js').read_text() + '</script></body>', 1)
    source = source.replace('</body>', '<script>' + (HERE / 'window_distribution.js').read_text() + '</script></body>', 1)
    (HERE.parent / 'index.html').write_text(source)
    numeric_values = {key: value for key, value in values.items() if not key.endswith(('_html', '_styles', '_figure', '_explorer'))}
    (RESULTS / 'article_values.json').write_text(json.dumps(numeric_values, indent=2) + '\n')
    print('Wrote article/index.html')


if __name__ == '__main__':
    main()
