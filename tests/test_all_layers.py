import pytest
import torch
from types import SimpleNamespace
from pathlib import Path

from analysis.large_study.all_layers import sequential_forward
from src.data import Window


class TwoLayerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])

    def forward(self, values, use_cache=False):
        first = self.layers[0](values)
        second = self.layers[1](first + torch.tensor([[[2., 0.]]]))
        return SimpleNamespace(logits=second)


def test_later_layer_selects_from_intervened_activations():
    model = TwoLayerModel()
    values = torch.tensor([[[1., 3.]]])
    calibrations = [{'means': torch.zeros(2), 'thresholds': {.5: {'N': threshold}}}
                    for threshold in (torch.tensor([2., 2.]), torch.tensor([2.5, 2.5]))]
    windows = [Window([0], 'window', 'article', 'test')]
    logits, statistics = sequential_forward(model, list(model.layers), calibrations, [torch.ones(2)] * 2,
                                             values, windows, 'gpt2', 'N', .5)
    # Layer 1 sees [2, 3] after the intervention and removes 2.
    # Its intact input would have been [3, 3], with nothing below 2.5.
    assert torch.equal(logits, torch.tensor([[[0., 3.]]]))
    assert statistics['observed_fraction'].item() == .5
    assert not model.layers[0]._forward_pre_hooks and not model.layers[1]._forward_pre_hooks


def test_hooks_are_removed_when_forward_fails():
    model = TwoLayerModel()
    with pytest.raises(ValueError):
        sequential_forward(model, list(model.layers), [{'means': torch.zeros(2), 'thresholds': {.5: {}}}] * 2,
                           [torch.ones(2)] * 2, torch.ones(1, 1, 2), [Window([0], 'w', 'a', 'test')], 'gpt2', 'N', .5)
    assert not model.layers[0]._forward_pre_hooks and not model.layers[1]._forward_pre_hooks


def test_identity_all_zero_and_random_control():
    model = TwoLayerModel()
    values = torch.tensor([[[1., 3.]]])
    calibrations = [{'means': torch.zeros(2), 'thresholds': {}}] * 2
    windows = [Window([0], 'window', 'article', 'test')]
    norms = [torch.ones(2)] * 2
    identity, identity_stats = sequential_forward(model, list(model.layers), calibrations, norms,
                                                   values, windows, 'gpt2', 'identity', 0.)
    zero, zero_stats = sequential_forward(model, list(model.layers), calibrations, norms,
                                           values, windows, 'gpt2', 'all_zero', 1.)
    random, random_stats = sequential_forward(model, list(model.layers), calibrations, norms,
                                               values, windows, 'gpt2', 'random', .5)
    repeated, _ = sequential_forward(model, list(model.layers), calibrations, norms,
                                      values, windows, 'gpt2', 'random', .5)
    assert torch.equal(identity, model(values).logits)
    assert torch.equal(zero, torch.zeros_like(zero))
    assert identity_stats['observed_fraction'].item() == 0
    assert zero_stats['observed_fraction'].item() == 1
    assert random_stats['observed_fraction'].item() == .5
    assert torch.equal(random, repeated)


def test_combine_sweeps_requires_a_complete_grid_and_matching_endpoints():
    import pandas as pd
    from analysis.expanded.methods import METHODS
    from analysis.large_study.combine_all_layers import combine_rows
    from analysis.large_study.evaluate import FRACTIONS, METRICS

    def sweep(fractions):
        conditions = [('identity', 0.), ('all_zero', 1.)]
        conditions += [(method, fraction) for fraction in (*fractions, 1.) for method in (*METHODS, 'random')]
        return pd.DataFrame([dict(model='gpt2', layer=-1, window_id='w', cluster_id='a',
                                  method=method, fraction=fraction, **{key: fraction for key in METRICS})
                             for method, fraction in conditions])
    initial, additional = sweep(FRACTIONS[:5]), sweep(FRACTIONS[5:])
    combined = combine_rows(initial, additional)
    assert len(combined) == 2 + (len(FRACTIONS) + 1) * (len(METHODS) + 1)
    with pytest.raises(ValueError, match='full fraction grid'):
        combine_rows(initial, additional.iloc[:-10])
    altered = additional.copy()
    altered.loc[altered.method == 'N', 'cluster_id'] = 'other'
    with pytest.raises(ValueError, match='inconsistent article'):
        combine_rows(initial, altered)
    additional.loc[additional.method == 'identity', 'kl'] = 1.
    with pytest.raises(ValueError, match='Endpoint measurements differ'):
        combine_rows(initial, additional)


def test_saved_all_layer_results_cover_the_single_layer_fraction_grid():
    import pandas as pd
    from analysis.expanded.methods import METHODS
    from analysis.large_study.evaluate import FRACTIONS

    root = Path(__file__).resolve().parents[1]
    for model in ('gpt2', 'smollm2'):
        data = pd.read_parquet(root / 'results/large_study/all_layers_combined' / model / 'metrics.parquet')
        assert data.window_id.nunique() == 64
        assert data.cluster_id.nunique() == 61
        for method in (*METHODS, 'random'):
            selected = data[data.method == method]
            assert set(selected.fraction) == {*FRACTIONS, 1.}
            assert selected.groupby('fraction').window_id.nunique().eq(64).all()
