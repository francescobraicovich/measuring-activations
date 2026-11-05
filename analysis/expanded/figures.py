#!/usr/bin/env python3
"""Create exploratory interactive figures from saved expanded-study window metrics."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
TARGETS = (("gpt2_6", "GPT-2, layer 6"), ("smollm2_16", "SmolLM2, layer 16"))
COLORS = {"N": "#0072B2", "L": "#D55E00", "G": "#009E73", "W": "#CC79A7"}


def window_values(table, method, fraction):
    selected = table[(table.method == method) & np.isclose(table.fraction, fraction)]
    if selected.empty or selected.duplicated(["window_id", "repetition"]).any():
        raise ValueError(f"Missing or duplicate observations: {method}, {fraction}")
    if not selected.groupby("window_id").cluster_id.nunique().eq(1).all():
        raise ValueError("A window has inconsistent article identifiers")
    return selected.groupby(["window_id", "cluster_id"], as_index=False)[["kl", "observed_fraction"]].mean()


def interval(values, replicates=1000):
    clusters = values.groupby("cluster_id").kl.agg(["sum", "count"])
    rng = np.random.default_rng(0)
    indices = rng.integers(0, len(clusters), size=(replicates, len(clusters)))
    estimates = clusters["sum"].to_numpy()[indices].sum(axis=1) / clusters["count"].to_numpy()[indices].sum(axis=1)
    return values.kl.mean(), *np.quantile(estimates, [.025, .975])


def plot_curve(figure, table, method, color, column, dash="solid", label=None, show_interval=True, visible=True):
    fractions = sorted(table.loc[table.method == method, "fraction"].unique())
    if not fractions:
        raise ValueError(f"Missing method {method}")
    windows = [window_values(table, method, fraction) for fraction in fractions]
    values = np.asarray([interval(window) for window in windows])
    observed = [window.observed_fraction.mean() for window in windows]
    label = label or method
    figure.add_trace(go.Scatter(x=observed, y=values[:, 0], customdata=fractions, mode="lines+markers", name=label,
        legendgroup=label, showlegend=column == 1, visible=visible, line=dict(color=color, dash=dash),
        error_y=dict(type="data", symmetric=False, array=values[:, 2] - values[:, 0],
                     arrayminus=values[:, 0] - values[:, 1], visible=show_interval),
        hovertemplate="Observed fraction: %{x:.3f}<br>Nominal fraction: %{customdata:.2f}<br>Mean KL: %{y:.5f}<extra>" + label + "</extra>"), row=1, col=column)


def base_figure(title):
    figure = make_subplots(rows=1, cols=2, subplot_titles=[item[1] for item in TARGETS])
    figure.update_layout(template="plotly_white", title=title, hovermode="closest",
        legend_title_text="Selection", height=680, margin=dict(l=80, r=30, t=100, b=150),
        annotations=list(figure.layout.annotations) + [dict(x=.5, y=-.27, xref="paper", yref="paper", showarrow=False, align="center",
            text="Exploratory; 1,000 article-cluster bootstrap replicates (four test articles per model).<br>"
                 "Independent y scales. Masking: all 128 positions; scoring: positions 16–126.")])
    figure.update_xaxes(title_text="Mean observed fraction removed", range=[0, 1])
    figure.update_yaxes(title_text="Mean KL (nats per scored token)")
    return figure


def add_references(figure, table, column):
    for method, fraction in (("identity", 0), ("all_zero", 1)):
        values = window_values(table, method, fraction)
        figure.add_trace(go.Scatter(x=[fraction], y=[values.kl.mean()], mode="markers", name=method.replace("_", " "),
            legendgroup=method, showlegend=column == 1, marker=dict(color="#666666", symbol="diamond", size=8),
            hovertemplate="Observed fraction: %{x:.2f}<br>Mean KL: %{y:.5f}<extra>" + method + "</extra>"), row=1, col=column)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/expanded/test")
    parser.add_argument("--output", type=Path, default=ROOT / "results/expanded/figures")
    args = parser.parse_args()
    tables = [pd.read_csv(args.input / label / "metrics.csv") for label, _ in TARGETS]
    for table in tables:
        if table.cluster_id.nunique() != 4:
            raise ValueError("Expected four held-out articles per model")
    args.output.mkdir(parents=True, exist_ok=True)
    main_figure = base_figure("Zero ablation: original rankings and matched random controls")
    centered_figure = base_figure("Appendix: original and mean-centered rankings, zero ablation")
    for column, table in enumerate(tables, 1):
        for method, color in COLORS.items():
            plot_curve(main_figure, table, method, color, column)
            plot_curve(main_figure, table, method + "_random", color, column, "dot", method + " random", False)
            plot_curve(centered_figure, table, method, color, column, label=method + " original")
            plot_curve(centered_figure, table, method + "_centered", color, column, "dash", method + " centered")
            plot_curve(centered_figure, table, method + "_centered_random", color, column, "dot",
                       method + " centered random", False, "legendonly")
        add_references(main_figure, table, column)
        add_references(centered_figure, table, column)
    for name, figure in (("original_rankings", main_figure), ("centered_rankings", centered_figure)):
        path = args.output / f"{name}.html"
        figure.write_html(path, include_plotlyjs=True)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
