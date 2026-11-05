"""Descriptive analyses of calibration activations; no held-out data is used."""
from itertools import combinations
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

METHODS = ("N", "L", "G", "W")


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def describe_activations(activations, norms, thresholds, fractions=(0, .25, .5, .75, .9, 1), *, zero_tolerance=1e-8, seed=0, density_sample=100000):
    """Summarize one calibration layer. Activations have shape [tokens, channels].

    N and G use strict threshold comparisons, except at the two endpoints.
    Density points are sampled; all mask and correlation summaries use all events.
    """
    a = np.asarray(activations)
    v = _array(norms).reshape(-1)
    if a.ndim != 2 or not a.shape[0] or a.shape[1] != len(v):
        raise ValueError("activations must have shape [tokens, channels] matching norms")
    if len(a) < 2:
        raise ValueError("at least two calibration tokens are required")
    if not np.isfinite(a).all() or not np.isfinite(v).all() or (v < 0).any():
        raise ValueError("activations and nonnegative norms must be finite")
    magnitude = np.abs(a)
    quantiles = np.quantile(magnitude, [.25, .5, .75, .9, .99, .999], axis=0)
    rms = np.sqrt(np.mean(np.square(a, dtype=np.float64), axis=0))
    split = len(a) // 2
    channels = pd.DataFrame({
        "channel": np.arange(a.shape[1]), "mean": a.mean(axis=0),
        "median": np.median(a, axis=0), "std": a.std(axis=0), "rms": rms,
        "negative_fraction": (a < 0).mean(axis=0), "positive_fraction": (a > 0).mean(axis=0),
        "zero_fraction": (magnitude <= zero_tolerance).mean(axis=0),
        "positive_q99": np.quantile(np.maximum(a, 0), .99, axis=0),
        "negative_q99": np.quantile(np.maximum(-a, 0), .99, axis=0),
        "weight_norm": v, "weighted_rms": rms * v,
        "tail_ratio": quantiles[4] / np.maximum(quantiles[1], zero_tolerance),
        "first_half_rms": np.sqrt(np.mean(np.square(a[:split], dtype=np.float64), axis=0)),
        "second_half_rms": np.sqrt(np.mean(np.square(a[split:], dtype=np.float64), axis=0)),
        "first_half_median_abs": np.median(magnitude[:split], axis=0),
        "second_half_median_abs": np.median(magnitude[split:], axis=0),
    })
    for q, values in zip(("25", "50", "75", "90", "99", "999"), quantiles):
        channels[f"abs_q{q}"] = values
    k = max(1, int(np.ceil(a.shape[1] * .1)))
    largest = np.partition(magnitude, -k, axis=1)[:, -k:]
    abs_sum = magnitude.sum(axis=1, dtype=np.float64)
    square_sum = np.square(a, dtype=np.float64).sum(axis=1)
    tokens = pd.DataFrame({"token": np.arange(len(a)), "abs_sum": abs_sum,
        "top_decile_abs_fraction": np.divide(largest.sum(axis=1), abs_sum, out=np.zeros(len(a)), where=abs_sum != 0),
        "top_decile_squared_fraction": np.divide(np.square(largest, dtype=np.float64).sum(axis=1), square_sum, out=np.zeros(len(a)), where=square_sum != 0)})
    for q in (.5, .75, .9, .99):
        tokens[f"abs_q{int(q*100)}"] = np.quantile(magnitude, q, axis=1)
    # Ranks use the average tie convention; mask ties instead use channel order.
    scores = {
        "N": (rankdata(magnitude, axis=0, method="average") / len(a)).astype(np.float32),
        "L": (rankdata(magnitude, axis=1, method="average") / a.shape[1]).astype(np.float32),
        "G": rankdata(magnitude.ravel(), method="average").reshape(a.shape) / a.size,
        "W": (rankdata(magnitude * v, axis=1, method="average") / a.shape[1]).astype(np.float32),
    }
    ordering = {"L": np.argsort(magnitude, axis=1, kind="stable"),
                "W": np.argsort(magnitude * v, axis=1, kind="stable")}
    rows, pairs, removals = [], [], []
    weighted = magnitude * v
    total_abs, total_sq, total_weighted = magnitude.sum(dtype=np.float64), square_sum.sum(), weighted.sum(dtype=np.float64)
    for p in fractions:
        if not 0 <= p <= 1 or p not in thresholds:
            raise ValueError(f"missing or invalid fraction: {p}")
        masks = {}
        for method in METHODS:
            if p == 0:
                mask = np.zeros(a.shape, dtype=bool)
            elif p == 1:
                mask = np.ones(a.shape, dtype=bool)
            elif method in ("N", "G"):
                mask = magnitude < _array(thresholds[p][method])
            else:
                mask = np.zeros(a.shape, dtype=bool)
                np.put_along_axis(mask, ordering[method][:, :int(p * a.shape[1])], True, axis=1)
            masks[method] = mask
            counts = mask.mean(axis=1)
            removals.append(pd.DataFrame({"token": np.arange(len(a)), "method": method, "fraction": p, "removed_fraction": counts}))
            rows.append({"method": method, "fraction": p, "observed_fraction": mask.mean(),
                "removed_abs_fraction": magnitude[mask].sum(dtype=np.float64) / total_abs if total_abs else 0,
                "removed_squared_fraction": np.square(a[mask], dtype=np.float64).sum() / total_sq if total_sq else 0,
                "removed_weighted_fraction": weighted[mask].sum(dtype=np.float64) / total_weighted if total_weighted else 0,
                "threshold_tie_fraction": np.mean(magnitude == _array(thresholds[p][method])) if method in ("N", "G") else np.nan})
            if p == .75:
                channels[f"removed_{method}"] = mask.mean(axis=0)
        if p == .75:
            channels["L_removed_N_kept"] = (masks["L"] & ~masks["N"]).mean(axis=0)
            channels["L_removed_W_kept"] = (masks["L"] & ~masks["W"]).mean(axis=0)
        for left, right in combinations(METHODS, 2):
            union = np.logical_or(masks[left], masks[right]).sum()
            pairs.append({"fraction": p, "left": left, "right": right,
                "jaccard": np.logical_and(masks[left], masks[right]).sum() / union if union else 1.,
                "disagreement": np.mean(masks[left] != masks[right])})
    correlations = []
    # Rank each full event vector once, retaining float64 for exact tie ranks.
    with TemporaryDirectory(prefix="activation_ranks_") as temporary:
        global_ranks = {}
        for method in METHODS:
            path = Path(temporary) / f"{method}.npy"
            rank_file = np.lib.format.open_memmap(path, mode="w+", dtype="float64", shape=(a.size,))
            rank_file[:] = rankdata(scores[method].ravel(), method="average")
            rank_file.flush()
            global_ranks[method] = rank_file
        for left, right in combinations(METHODS, 2):
            # Accumulate centered products in chunks to avoid a full stacked copy.
            means = [(a.size + 1) / 2] * 2
            sums = np.zeros(3, dtype=np.float64)
            for start in range(0, a.size, 1_000_000):
                x = global_ranks[left][start:start + 1_000_000] - means[0]
                y = global_ranks[right][start:start + 1_000_000] - means[1]
                sums += [np.dot(x, y), np.dot(x, x), np.dot(y, y)]
            coefficient = sums[0] / np.sqrt(sums[1] * sums[2]) if sums[1] and sums[2] else np.nan
            correlations.append({"left": left, "right": right, "spearman": coefficient})
        del global_ranks, rank_file
    rng = np.random.default_rng(seed)
    indices = rng.choice(a.size, min(density_sample, a.size), replace=False)
    density = pd.DataFrame({method: scores[method].ravel()[indices] for method in METHODS})
    # Deterministic calibration examples at low/median/high quantiles of each statistic.
    example_ids = set()
    for column in ("rms", "tail_ratio", "weight_norm"):
        order = np.argsort(channels[column].to_numpy(), kind="stable")
        example_ids.update(order[int(q * (len(order)-1))] for q in (.1, .5, .9))
    examples = pd.DataFrame({"token": np.tile(np.arange(len(a)), len(example_ids)),
        "channel": np.repeat(sorted(example_ids), len(a)), "activation": a[:, sorted(example_ids)].T.ravel()})
    return {"channels": channels, "tokens": tokens, "removal": pd.DataFrame(rows),
            "token_removal": pd.concat(removals, ignore_index=True), "overlap": pd.DataFrame(pairs),
            "correlation": pd.DataFrame(correlations), "density": density, "examples": examples,
            "thresholds": pd.DataFrame([{ "fraction": p, "global": float(_array(thresholds[p]["G"]))} for p in fractions])}
