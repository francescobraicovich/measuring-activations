"""Compute paired KL contrasts with article and window bootstrap intervals."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
metrics = pd.read_parquet(ROOT / 'results/study/metrics.parquet')
rng = np.random.default_rng(1729)
rows = []
for (model, layer, fraction), group in metrics.groupby(['model', 'layer', 'fraction']):
    for left, right in [('W', 'L'), ('N', 'G')]:
        a, b = (group[group.method == method] for method in (left, right))
        if a.empty or b.empty:
            continue
        pair = a.merge(b, on=['window_id', 'cluster_id'], suffixes=('_left', '_right'), validate='one_to_one')
        if len(pair) != len(a) or len(pair) != len(b):
            raise ValueError('paired windows do not match')
        delta = pair.kl_left.to_numpy() - pair.kl_right.to_numpy()
        row = dict(model=model, layer=layer, fraction=fraction, comparison=f'{left} minus {right}',
                   kl_difference=delta.mean(), left_removal=pair.observed_fraction_left.mean(),
                   right_removal=pair.observed_fraction_right.mean(), windows=len(pair),
                   articles=pair.cluster_id.nunique())
        for name, clusters in [('article', pair.cluster_id), ('window', pair.window_id)]:
            sums = pd.DataFrame({'cluster': clusters, 'delta': delta}).groupby('cluster').delta.agg(['sum', 'count'])
            indexes = rng.integers(0, len(sums), size=(1000, len(sums)))
            estimates = sums['sum'].to_numpy()[indexes].sum(axis=1) / sums['count'].to_numpy()[indexes].sum(axis=1)
            row[name + '_low'], row[name + '_high'] = np.quantile(estimates, [.025, .975])
        rows.append(row)
pd.DataFrame(rows).to_csv(ROOT / 'results/study/paired_method_comparisons.csv', index=False)
