"""Publication figures from saved ablation and projection-geometry measurements."""
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
COLORS = {'L': '#26736d', 'W': '#ad6540', 'random': '#7b8089', 'W_centered': '#70618c'}


def save(figure, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ('svg', 'png'):
        figure.savefig(OUT / f'{name}.{extension}', dpi=190, facecolor='#fcfbf8', bbox_inches='tight')
    plt.close(figure)


def build_figures():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.labelsize': 11,
                         'axes.titlesize': 13, 'axes.titleweight': 'medium', 'text.color': '#243735',
                         'axes.labelcolor': '#243735', 'xtick.color': '#56615e', 'ytick.color': '#56615e',
                         'axes.edgecolor': '#c2c8c2', 'axes.facecolor': '#fcfbf8',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'mathtext.fontset': 'cm',
                         'svg.fonttype': 'none', 'svg.hashsalt': 'activation-strength-report'})
    summary = pd.read_csv(ROOT / 'results/large_study/data/summary.csv')
    selected = summary[(summary.fraction == .75) & (summary.metric == 'kl') & summary.method.isin(['L', 'W', 'random'])]
    if len(selected) != 30 or (selected.lower <= 0).any():
        raise ValueError('Expected positive KL measurements for three methods at ten layers')
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharey=True, layout='constrained')
    for ax, model, title in zip(axes, ['smollm2', 'gpt2'], ['SmolLM2-135M', 'GPT-2 Small']):
        group = selected[selected.model == model]
        for method, label, marker in [('random', 'Random', '^'), ('W', 'W · contribution norm', 's'), ('L', 'L · magnitude', 'o')]:
            part = group[group.method == method].sort_values('layer')
            ax.errorbar(part.layer, part['mean'], yerr=[part['mean'] - part.lower, part.upper - part['mean']],
                        color=COLORS[method], marker=marker, markersize=5, linewidth=1.5,
                        linestyle='--' if method == 'random' else '-', capsize=2.5, label=label)
        ax.set_yscale('log')
        ax.set_xticks(sorted(group.layer.unique()))
        ax.set_xlabel('Layer index (zero-based)', labelpad=10)
        ax.set_title(title, loc='left', pad=14)
        ax.grid(axis='y', which='major', color='#e0e3dc', linewidth=.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='both', length=3)
    axes[0].set_ylabel('KL from intact model (nats / scored token)', labelpad=12)
    axes[0].legend(frameon=False, loc='upper right', fontsize=9)
    axes[1].annotate(r'$\mathrm{KL}_{W}>\mathrm{KL}_{L}$', xy=(0, .814), xytext=(2.4, 2.5), fontsize=10,
                     arrowprops={'arrowstyle': '-', 'color': '#ad6540', 'linewidth': .9})
    contrasts = pd.read_csv(ROOT / 'results/large_study/data/paired_comparisons.csv')
    reversal = contrasts[(contrasts.model == 'gpt2') & (contrasts.layer == 6) &
                         (contrasts.method == 'L') & (contrasts.reference == 'random') &
                         (contrasts.fraction == .75) & (contrasts.metric == 'kl')]
    if len(reversal) != 1:
        raise ValueError('Missing reversal annotation')
    reversal_mean = selected[(selected.model == 'gpt2') & (selected.layer == 6) & (selected.method == 'L')]['mean'].item()
    axes[1].annotate(r'$\mathrm{KL}_{L}-\mathrm{KL}_{\mathrm{random}} = ' + f'{reversal.iloc[0]["mean"]:+.2g}' + '$', xy=(6, reversal_mean), xytext=(3.8, .25), fontsize=9,
                     arrowprops={'arrowstyle': '-', 'color': '#26736d', 'linewidth': .8})
    save(fig, 'ablation_comparison')

    geometry = pd.read_parquet(ROOT / 'results/followup/localization/geometry.parquet')
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.2), layout='constrained')
    for ax, field, title in zip(axes, ['sum_individual_norms', 'net_removed_norm'],
                                ['Sum of individual lengths', 'Length after summing vectors']):
        for index, method in enumerate(['L', 'W', 'W_centered']):
            group = geometry[(geometry.method == method) & (geometry.region == 'scored')]
            clusters = group.groupby('cluster_id')[field].agg(['sum', 'count'])
            if len(group) != 128 or len(clusters) != 4:
                raise ValueError('Unexpected geometry sample')
            rng = np.random.default_rng(1729)
            indices = rng.integers(0, len(clusters), size=(1000, len(clusters)))
            draws = clusters['sum'].to_numpy()[indices].sum(1) / clusters['count'].to_numpy()[indices].sum(1)
            low, high = np.quantile(draws, [.025, .975])
            mean = group[field].mean()
            ax.barh(index, mean, color=COLORS[method], height=.48)
            ax.errorbar(mean, index, xerr=[[mean-low], [high-mean]], color='#243735', capsize=3, linewidth=.9)
            ax.text(high + (9 if field == 'sum_individual_norms' else .3), index, f'{mean:.3g}', va='center', fontsize=10)
        ax.set_yticks(range(3), ['L', 'W', 'W centered'])
        ax.invert_yaxis()
        ax.set_xlim(0, ax.get_xlim()[1] * 1.13)
        ax.set_xlabel('Mean length (internal projection units)', labelpad=10)
        ax.set_title(title, loc='left', pad=14)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', color='#e0e3dc', linewidth=.7)
        ax.set_axisbelow(True)
    save(fig, 'projection_geometry')


if __name__ == '__main__':
    build_figures()
