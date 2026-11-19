"""Publication figures from saved ablation measurements."""
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/measuring-activations-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/large_study/figures'
COLORS = {'N': '#d72000', 'G': '#253367', 'L': '#ee6100', 'W': '#1bb6af', 'random': '#000000', 'L_centered': '#ffad0a', 'W_centered': '#1bb6af'}


def save(figure, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ('svg', 'png'):
        figure.savefig(OUT / f'{name}.{extension}', metadata={'Date': None} if extension == 'svg' else None,
                       dpi=190, facecolor='#ffffff', bbox_inches='tight')
    plt.close(figure)


def window_flip_rates():
    """Load the held-out flip rates for the SmolLM2 example."""
    table = pd.read_parquet(OUT.parent / 'data/metrics.parquet',
                            columns=['model', 'layer', 'method', 'fraction', 'flip_rate'])
    selected = table[(table.model == 'smollm2') & (table.layer == 16) &
                     (table.method == 'G') & (table.fraction == .75)]
    if len(selected) != 512 or selected.flip_rate.isna().any():
        raise ValueError('Unexpected sample for window-level flip rates')
    return selected.flip_rate


def window_distributions():
    """Summarize the per-window outcomes for each measured setting for the four definitions."""
    metrics = {
        'flip_rate': 'Next-token flip rate (%)',
        'kl': 'KL divergence (nats per scored token)',
        'nll_delta': 'NLL change (nats per scored token)',
    }
    definitions = {
        'N': 'N · Neuron percentile',
        'L': 'L · Within-token percentile',
        'G': 'G · Pooled layer percentile',
        'W': 'W · Output-weighted percentile',
    }
    columns = ['model', 'layer', 'method', 'fraction', *metrics]
    results = OUT.parent
    paths = [results / 'data/metrics.parquet', *(
        results / 'all_layers_combined' / model / 'metrics.parquet' for model in ('gpt2', 'smollm2')
    )]
    table = pd.concat(pd.read_parquet(path, columns=columns, filters=[('method', 'in', list(definitions))]) for path in paths)
    models = {}
    for (model, layer, fraction, method), windows in table.groupby(['model', 'layer', 'fraction', 'method']):
        outcomes = {}
        for metric in metrics:
            values = windows[metric].dropna()
            if metric == 'flip_rate':
                values = values * 100
            bins = range(max(1, math.ceil(values.max())) + 1) if metric == 'flip_rate' else 12
            counts, edges = np.histogram(values, bins=bins)
            markers = [values.median(), values.quantile(.9), values.max()]
            outcomes[metric] = [edges.tolist(), counts.tolist(), markers]
        fractions = models.setdefault(model, {}).setdefault(str(layer), {})
        fractions.setdefault(str(fraction), {})[method] = outcomes
    return {'models': models, 'metrics': metrics, 'definitions': definitions,
            'colors': {method: COLORS[method] for method in definitions}}


def plot_window_flip_rates(rates):
    percentages = rates * 100
    maximum = math.ceil(percentages.max())
    figure, axis = plt.subplots(figsize=(8.2, 3.6), layout='constrained')
    axis.hist(percentages, bins=range(maximum + 1), color=COLORS['G'], alpha=.22, edgecolor=COLORS['G'])
    for label, value, style in (
        ('Median', percentages.median(), '-'),
        ('90th percentile', percentages.quantile(.9), '--'),
        ('Maximum', percentages.max(), ':'),
    ):
        axis.axvline(value, color=COLORS['G'], linestyle=style, linewidth=1.6, label=f'{label} · {value:.1f}%')
    axis.set(xlim=(0, maximum + .25), xticks=range(0, maximum + 1, 2), xlabel='Next-token flip rate per window (%)',
             ylabel='Number of windows')
    axis.set_ylim(bottom=0)
    axis.grid(axis='y', color='#e0e3dc', linewidth=.7)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc='lower center', bbox_to_anchor=(.5, 1), ncol=3)
    save(figure, 'window_flip_rates')


def build_figures():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.labelsize': 11,
                         'axes.titlesize': 13, 'axes.titleweight': 'medium', 'text.color': '#000000',
                         'axes.labelcolor': '#000000', 'xtick.color': '#000000', 'ytick.color': '#000000',
                         'axes.edgecolor': '#c2c8c2', 'axes.facecolor': '#ffffff',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'mathtext.fontset': 'cm',
                         'svg.fonttype': 'none', 'svg.hashsalt': 'activation-strength-report'})
    summary = pd.read_csv(ROOT / 'results/large_study/data/summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True, layout='constrained')
    for ax, model, layer, title in zip(axes, ['gpt2', 'smollm2'], [6, 16], ['GPT-2 · layer 6', 'SmolLM2 · layer 16']):
        for method, label, linestyle in [('random', 'Random', '--'), ('G', 'G · pooled percentile', '-'), ('N', 'N · neuron percentile', '-')]:
            selection = summary[(summary.model == model) & (summary.layer == layer) & (summary.method == method)]
            outcome = selection[selection.metric == 'flip_rate']
            observed = selection[selection.metric == 'observed_fraction'][['fraction', 'mean']]
            part = outcome.merge(observed, on='fraction', suffixes=('', '_observed')).sort_values('fraction')
            if len(part) != 16:
                raise ValueError(f'Unexpected removal curve for {model} {method}')
            ax.plot([0, *part.mean_observed * 100], [0, *part['mean'] * 100], color=COLORS[method],
                    linewidth=1.8, linestyle=linestyle, label=label)
            ax.fill_between(part.mean_observed * 100, part.lower * 100, part.upper * 100,
                            color=COLORS[method], alpha=.12, linewidth=0)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 26)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel('Mean activations removed (%)', labelpad=10)
        ax.set_title(title, loc='left', pad=12)
        ax.grid(axis='y', color='#e0e3dc', linewidth=.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Top-token predictions changed (%)', labelpad=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='upper center', bbox_to_anchor=(.5, -.02), ncol=3, fontsize=10)
    fig.suptitle('Middle-layer token flip rates under weak and random removal', x=.02, ha='left', fontsize=15)
    save(fig, 'ablation_comparison')
    plot_window_flip_rates(window_flip_rates())


if __name__ == '__main__':
    build_figures()
