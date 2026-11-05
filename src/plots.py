"""Interactive figures generated exclusively from saved experiment tables."""
from pathlib import Path

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

COLORS = {"N": "#0072B2", "L": "#D55E00", "G": "#009E73", "W": "#CC79A7", "N_random": "#0072B2", "L_random": "#D55E00", "G_random": "#009E73", "W_random": "#CC79A7", "all_zero": "#222222", "identity": "#999999"}


def _matrix(table, value, diagonal):
    methods = ["N", "L", "G", "W"]
    matrix = pd.DataFrame(np.eye(4) * diagonal, index=methods, columns=methods)
    for row in table.itertuples():
        matrix.loc[row.left, row.right] = matrix.loc[row.right, row.left] = getattr(row, value)
    return matrix


def descriptive_figures(directory):
    """Read the tables produced by describe_activations and return Plotly figures."""
    directory = Path(directory)
    tables = {name: pd.read_csv(directory / f"{name}.csv") for name in
              ("channels", "tokens", "removal", "token_removal", "overlap", "correlation", "density", "examples")}
    channels = tables["channels"]
    figures = {}
    examples = tables["examples"].copy()
    examples["channel"] = examples["channel"].astype(str)
    figures["signed_distributions"] = px.histogram(examples, x="activation", color="channel", barmode="overlay", opacity=.55, marginal="box", title="Calibration: signed activation distributions")
    examples["log_magnitude"] = np.log10(np.abs(examples.activation) + 1e-8)
    figures["magnitude_distributions"] = px.histogram(examples, x="log_magnitude", color="channel", barmode="overlay", opacity=.55, title="Calibration: log10 absolute activation (+1e-8)")
    figures["channel_rms"] = px.histogram(channels, x="rms", log_x=True, title="Distribution of channel RMS")
    valid = channels[(channels.rms > 0) & (channels.weight_norm > 0)].copy()
    correlation = spearmanr(channels.rms, channels.weight_norm).statistic
    fig = px.scatter(valid, x="rms", y="weight_norm", color="positive_fraction", hover_data=["channel", "weighted_rms"], log_x=True, log_y=True, title=f"RMS and outgoing weight norm (Spearman ρ={correlation:.3f})")
    if len(valid) >= 10:
        valid["bin"] = pd.qcut(valid.rms, min(10, len(valid)), duplicates="drop")
        trend = valid.groupby("bin", observed=True)[["rms", "weight_norm"]].median()
        fig.add_scatter(x=trend.rms, y=trend.weight_norm, mode="lines+markers", name="Binned median", line={"color": "black"})
    figures["scale_weight"] = fig
    figures["weighted_scale"] = px.histogram(channels, x="weighted_rms", log_x=True, title="Typical output-weighted scale")
    overlap = tables["overlap"]
    overlap = overlap[np.isclose(overlap.fraction, .75)]
    for name, frame, value, title in (("jaccard", overlap, "jaccard", "Mask Jaccard at p=0.75"), ("rank_correlation", tables["correlation"], "spearman", "Score rank correlation")):
        matrix = _matrix(frame, value, 1.)
        figures[name] = px.imshow(matrix, zmin=-1 if value == "spearman" else 0, zmax=1, color_continuous_scale="Viridis", text_auto=".2f", title=title)
    for left, right in (("N", "L"), ("L", "W")):
        figures[f"density_{left}_{right}"] = px.density_heatmap(tables["density"], x=left, y=right, nbinsx=50, nbinsy=50, title=f"{left} versus {right} percentile scores (sampled events)")
    removal = tables["removal"].melt(id_vars=["method", "fraction"], value_vars=["observed_fraction", "removed_abs_fraction", "removed_squared_fraction", "removed_weighted_fraction"], var_name="measure", value_name="value")
    figures["removal"] = px.line(removal, x="fraction", y="value", color="method", color_discrete_map=COLORS, facet_col="measure", facet_col_wrap=2, markers=True, title="Observed removals and removed signal")
    token_removal = tables["token_removal"]
    token_removal = token_removal[np.isclose(token_removal.fraction, .75)]
    figures["token_removal"] = px.violin(token_removal, x="method", y="removed_fraction", color="method", color_discrete_map=COLORS, box=True, title="Per-token removal at p=0.75")
    scales = channels.melt(id_vars=["channel"], value_vars=["rms", "weight_norm", "weighted_rms"], var_name="measure", value_name="value")
    figures["scale_distributions"] = px.histogram(scales[scales.value > 0], x="value", facet_col="measure", facet_col_wrap=3, log_x=True, title="Channel scale and outgoing-weight distributions")
    figures["token_concentration"] = px.histogram(tables["tokens"].melt(id_vars="token", value_vars=["top_decile_abs_fraction", "top_decile_squared_fraction"], var_name="measure", value_name="value"), x="value", color="measure", barmode="overlay", opacity=.6, title="Fraction of token signal in the largest 10% of channels")
    thresholds_path = directory / "thresholds.csv"
    if thresholds_path.exists():
        thresholds = pd.read_csv(thresholds_path)
        threshold75 = thresholds[np.isclose(thresholds.fraction, .75)]
        if not threshold75.empty:
            cutoffs = pd.concat([pd.DataFrame({"type": "L: token p75", "cutoff": tables["tokens"].abs_q75}), pd.DataFrame({"type": "N: channel p75", "cutoff": channels.abs_q75})])
            fig = px.histogram(cutoffs, x="cutoff", color="type", barmode="overlay", opacity=.55, title="Calibration p75 cutoffs")
            fig.add_vline(x=float(threshold75["global"].iloc[0]), line_dash="dash", annotation_text="G pooled p75")
            figures["thresholds"] = fig
    removal_columns = [f"removed_{method}" for method in ("N", "L", "G", "W") if f"removed_{method}" in channels]
    if removal_columns:
        rates = channels.melt(id_vars=["channel", "rms", "weight_norm"], value_vars=removal_columns, var_name="method", value_name="removal_rate")
        rates["method"] = rates.method.str.removeprefix("removed_")
        figures["channel_removal"] = px.scatter(rates[rates.rms > 0], x="rms", y="removal_rate", color="method", color_discrete_map=COLORS, facet_col="method", hover_data=["channel", "weight_norm"], log_x=True, title="Per-channel removal rate at p=0.75")
        if {"removed_N", "removed_L", "removed_W"}.issubset(channels.columns):
            disagreement = channels.copy()
            disagreement["L_minus_N"] = disagreement.removed_L - disagreement.removed_N
            disagreement["L_minus_W"] = disagreement.removed_L - disagreement.removed_W
            figures["channel_disagreement"] = px.scatter(disagreement[(disagreement.rms > 0) & (disagreement.weight_norm > 0)], x="rms", y="weight_norm", color="L_minus_W", hover_data=["channel", "L_minus_N", "removed_N", "removed_L", "removed_W", "L_removed_N_kept", "L_removed_W_kept"], log_x=True, log_y=True, color_continuous_scale="RdBu", color_continuous_midpoint=0, title="Per-channel removal-rate difference L minus W")
    return figures


def descriptive_summary(directory):
    """Create a concise, calibration-only summary from saved tables."""
    directory = Path(directory)
    channels = pd.read_csv(directory / "channels.csv")
    removal = pd.read_csv(directory / "removal.csv")
    overlap = pd.read_csv(directory / "overlap.csv")
    meta_path = directory / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    rng = np.random.default_rng(0)
    correlations = []
    for _ in range(1000):
        sample = rng.integers(len(channels), size=len(channels))
        correlations.append(spearmanr(channels.rms.iloc[sample], channels.weight_norm.iloc[sample]).statistic)
    correlation = spearmanr(channels.rms, channels.weight_norm).statistic
    low, high = np.nanquantile(correlations, [.025, .975])
    half_rms = spearmanr(channels.first_half_rms, channels.second_half_rms).statistic
    half_median = spearmanr(channels.first_half_median_abs, channels.second_half_median_abs).statistic
    contribution = spearmanr(channels.rms, channels.weighted_rms).statistic
    count = max(1, int(np.ceil(len(channels) * .1)))
    top_rms = set(channels.nlargest(count, "rms").channel)
    top_weighted = set(channels.nlargest(count, "weighted_rms").channel)
    selected = removal[np.isclose(removal.fraction, .75)]
    lines = ["# Calibration descriptive summary", "", f"Model: {metadata.get('model', 'unspecified')}; layer: {metadata.get('layer', 'unspecified')}; channels: {len(channels)}.",
             f"RMS versus outgoing norm: Spearman {correlation:.3f} (channel bootstrap 95% interval {low:.3f} to {high:.3f}).",
             f"Half-sample scale rank stability: RMS {half_rms:.3f}; median absolute activation {half_median:.3f}.",
             f"RMS versus weighted RMS rank correlation: {contribution:.3f}; top-decile overlap: {len(top_rms & top_weighted) / count:.3f}.",
             f"Median negative fraction: {channels.negative_fraction.median():.3f}; median effectively zero fraction (|a| ≤ 1e-8): {channels.zero_fraction.median():.3f}.", "", "At nominal p=0.75:", "", "| Method | Observed fraction | Removed absolute mass | Removed squared mass | Removed weighted mass | Threshold tie fraction |", "|---|---:|---:|---:|---:|---:|"]
    for row in selected.itertuples():
        tie = f"{row.threshold_tie_fraction:.3f}" if np.isfinite(row.threshold_tie_fraction) else "—"
        lines.append(f"| {row.method} | {row.observed_fraction:.3f} | {row.removed_abs_fraction:.3f} | {row.removed_squared_fraction:.3f} | {row.removed_weighted_fraction:.3f} | {tie} |")
    lines.extend(["", "Mask Jaccard at p=0.75: " + "; ".join(f"{row.left}–{row.right} {row.jaccard:.3f}" for row in overlap[np.isclose(overlap.fraction, .75)].itertuples()) + ".", "", "Calibration-only description; no causal conclusions. L and G have the same within-token magnitude ordering but different removal budgets. L and W minimize their respective local proxy masses by construction; these masses are not independent causal evidence."])
    return "\n".join(lines) + "\n"


def causal_figures(metrics, paired=None):
    """Plot saved per-window metrics; points are window-weighted means."""
    metrics = pd.read_csv(metrics) if isinstance(metrics, (str, Path)) else metrics.copy()
    required = {"model", "layer", "method", "fraction", "observed_fraction", "removed_abs_fraction", "removed_weighted_fraction", "kl"}
    if not required.issubset(metrics.columns):
        raise ValueError(f"metrics missing columns: {sorted(required - set(metrics.columns))}")
    grouping = ["model", "layer", "method", "fraction"]
    averaged = metrics.groupby(grouping, as_index=False)[["observed_fraction", "removed_abs_fraction", "removed_weighted_fraction", "kl"]].mean()
    if "cluster_id" in metrics.columns:
        rng = np.random.default_rng(0)
        intervals = []
        for key, group in metrics.groupby(grouping, dropna=False):
            clusters = [part.kl.to_numpy() for _, part in group.groupby("cluster_id")]
            samples = []
            for _ in range(1000):
                chosen = rng.integers(len(clusters), size=len(clusters))
                samples.append(np.concatenate([clusters[i] for i in chosen]).mean())
            low, high = np.quantile(samples, [.025, .975])
            intervals.append((*key, low, high))
        intervals = pd.DataFrame(intervals, columns=grouping + ["low", "high"])
        averaged = averaged.merge(intervals, on=grouping)
        averaged["error_plus"] = (averaged.high - averaged.kl).clip(lower=0)
        averaged["error_minus"] = (averaged.kl - averaged.low).clip(lower=0)
    figures = {}
    for axis, name in (("observed_fraction", "causal_events"), ("removed_abs_fraction", "causal_absolute_mass"), ("removed_weighted_fraction", "causal_weighted_mass")):
        figures[name] = px.line(averaged.sort_values(axis), x=axis, y="kl", color="method", color_discrete_map=COLORS, line_dash="method", line_group="method", facet_row="model", facet_col="layer", markers=True, error_y="error_plus" if "error_plus" in averaged else None, error_y_minus="error_minus" if "error_minus" in averaged else None, title="Mean KL by observed removal" if axis == "observed_fraction" else f"Mean KL by {axis.replace('_', ' ')}")
    if "depth" in metrics.columns:
        depth = metrics[(np.isclose(metrics.fraction, .75)) | (metrics.method == "all_zero")]
        if not depth.empty:
            means = depth.groupby(["model", "depth", "method"], as_index=False).kl.mean()
            figures["depth"] = px.bar(means, x="depth", y="kl", color="method", color_discrete_map=COLORS, facet_col="model", barmode="group", category_orders={"depth": ["early", "middle", "final"]}, title="KL at p=0.75 and all-zero reference by depth")
    if paired is not None:
        paired = pd.read_csv(paired) if isinstance(paired, (str, Path)) else paired.copy()
        if not paired.empty:
            paired["error_plus"] = (paired.upper - paired.paired_kl_difference).clip(lower=0)
            paired["error_minus"] = (paired.paired_kl_difference - paired.lower).clip(lower=0)
            figures["paired_contrasts"] = px.scatter(paired, x="fraction", y="paired_kl_difference", color="method", color_discrete_map=COLORS, facet_row="model", facet_col="layer", error_y="error_plus", error_y_minus="error_minus", title="Paired KL difference: method minus count-matched random")
            figures["paired_contrasts"].add_hline(y=0, line_dash="dash")
    return figures


def write_figures(figures, destination):
    """Write self-contained interactive HTML files."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        figure.write_html(destination / f"{name}.html", include_plotlyjs=True)
