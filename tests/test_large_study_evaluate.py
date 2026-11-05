import math

import numpy as np
import torch

from analysis.expanded.methods import METHODS, build_mask, fit_calibration
from analysis.large_study.evaluate import FRACTIONS, make_mask, prepare_scores, summarize
from src.data import Window


def test_cached_masks_match_direct_masks():
    values = torch.tensor([[[1., -1., 0., 3.], [2., 2., -4., 1.]]])
    calibration = fit_calibration(np.array([[1., -1., 0., 3.], [2., 2., -4., 1.], [0., 3., 1., 4.]], dtype=np.float32), fractions=(0., *FRACTIONS, 1.))
    norms = torch.tensor([1., 2., 3., 4.])
    absolute, centered, ranks = prepare_scores(values, norms, calibration['means'])
    windows = [Window([0, 1], 'window', 'article', 'test')]
    for fraction in FRACTIONS:
        for method in METHODS:
            actual = make_mask(method, fraction, absolute, centered, ranks, calibration, windows, 'gpt2', 0)
            expected = build_mask(values, method, fraction, calibration['thresholds'][fraction], norms, calibration['means'])
            assert torch.equal(actual, expected)
            if method.startswith(('L', 'W')):
                assert (actual.sum(-1) == math.floor(fraction * values.shape[-1])).all()


def test_article_bootstrap_is_finite():
    from analysis.large_study.evaluate import METRICS
    rows = [dict(model='gpt2', layer=0, method='L', fraction=.5, cluster_id=f'a{i // 2}', **{key: float(i) for key in METRICS}) for i in range(4)]
    summary = summarize(rows, replicates=100)
    assert len(summary) == len(METRICS)
    assert summary.clusters.eq(2).all()
    assert np.isfinite(summary[['mean', 'lower', 'upper']].to_numpy()).all()
