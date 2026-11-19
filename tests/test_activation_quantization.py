from types import SimpleNamespace

import numpy as np
import pytest
import torch

from analysis.activation_quantization.quantization import (
    ActivationQuantizer, fit_codebooks, positive_levels, quantized_forward, reconstruct,
)


def test_sd_band_interval_counts_and_equal_precision_budget():
    geometric = positive_levels('geometric', 4, sigma=1., clip=3.)
    assert np.allclose(geometric, [0, 1, 1.5, 2, 2.25, 2.5, 2.75, 3])
    for bits in (4, 6, 8):
        for clip in (0.1, 1., 2.5, 3., 1000.):
            levels = positive_levels('geometric', bits, 1., clip)
            uniform = positive_levels('uniform', bits, 1., clip)
            assert len(levels) == len(uniform) == 2 ** (bits - 1)
            assert levels[0] == 0 and levels[-1] == clip
            assert np.all(np.diff(levels) > 0)


def test_nearest_levels_ties_sign_zero_and_saturation():
    levels = np.array([0., 1., 1.5, 2.])
    values = np.array([-100, -2, -1.75, -1.26, -1.25, -.5, 0., .5, .51, 1.25, 1.26, 100.])
    expected = np.array([-2, -2, -1.5, -1.5, -1, 0., 0., 0., 1., 1., 1.5, 2.])
    quantizer = ActivationQuantizer(levels)
    actual = quantizer(torch.tensor(values.reshape(2, 2, 3), dtype=torch.float32))
    assert actual.shape == (2, 2, 3)
    np.testing.assert_allclose(actual.numpy().reshape(-1), expected)
    np.testing.assert_allclose(reconstruct(values, levels), expected)


def test_uniform_rounding_has_the_expected_error_bound_inside_range():
    levels = positive_levels('uniform', 4, 1., 7.)
    values = np.linspace(-7., 7., 10001)
    assert np.max(np.abs(reconstruct(values, levels) - values)) <= .5 + 1e-12


def test_clipping_selection_uses_the_minimum_training_mse_and_shared_range():
    sample = np.concatenate((np.linspace(-.5, .5, 1000), [20.]))
    fitted, candidates = fit_codebooks(sample, sigma=1., maximum=20., bits=4)
    for method, kind in [('uniform_mse', 'uniform'), ('g_geometric', 'geometric')]:
        assert fitted[method]['calibration_mse'] == min(record['mse'] for record in candidates if record['kind'] == kind)
    assert fitted['uniform_matched']['clip'] == fitted['g_geometric']['clip']
    assert fitted['uniform_max']['clip'] == 20.
    assert {record['level_count'] for record in fitted.values()} == {15}
    assert set(record['clip'] for record in candidates if record['kind'] == 'uniform') == set(record['clip'] for record in candidates if record['kind'] == 'geometric')


@pytest.mark.parametrize('arguments', [('unknown', 4, 1., 3.), ('uniform', 1, 1., 3.),
                                      ('uniform', 4, 0., 3.), ('uniform', 4, 1., float('inf'))])
def test_invalid_codebook_parameters_are_rejected(arguments):
    with pytest.raises(ValueError):
        positive_levels(*arguments)


@pytest.mark.parametrize('sample', [[], [float('nan')], [float('inf')]])
def test_invalid_calibration_samples_are_rejected(sample):
    with pytest.raises(ValueError, match='sample'):
        fit_codebooks(sample, 1., 3., 4)


@pytest.mark.parametrize('levels', [[1., 2.], [0., 1., 1.], [0., float('nan')], [0.]])
def test_invalid_reconstruction_levels_are_rejected(levels):
    with pytest.raises(ValueError, match='levels'):
        ActivationQuantizer(levels)


class TwoProjectionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])

    def forward(self, values, use_cache=False):
        return SimpleNamespace(logits=self.layers[1](self.layers[0](values) + .4))


def test_sequential_quantization_uses_current_inputs_and_removes_hooks():
    model = TwoProjectionModel()
    values = torch.tensor([[[.4, 1.4]]])
    quantizers = [ActivationQuantizer([0., 1., 2.])] * 2
    actual = quantized_forward(model, model.layers, quantizers, values)
    assert torch.equal(actual, torch.tensor([[[0., 1.]]]))
    assert not any(layer._forward_pre_hooks for layer in model.layers)
    assert torch.allclose(model(values).logits, values + .4)


def test_hooks_are_removed_after_failure_and_lengths_are_checked():
    model = TwoProjectionModel()
    def fail(_values):
        raise RuntimeError('deliberate test failure')
    with pytest.raises(RuntimeError, match='deliberate'):
        quantized_forward(model, model.layers, [lambda value: value, fail], torch.ones(1, 1, 2))
    assert not any(layer._forward_pre_hooks for layer in model.layers)
    with pytest.raises(ValueError, match='one quantizer'):
        quantized_forward(model, model.layers, [], torch.ones(1, 1, 2))


def test_paired_article_intervals_keep_window_weights_and_pairing():
    import pandas as pd
    from analysis.activation_quantization.report import article_intervals, paired_comparisons

    records = []
    for method in ('g_geometric', 'uniform_max', 'uniform_mse', 'uniform_matched'):
        for index, article in enumerate(('a', 'a', 'b')):
            reference = (1., 1., 2.)[index]
            difference = (1., 1., 4.)[index] if method == 'g_geometric' else 0.
            records.append(dict(model='gpt2', scope='all_layers', bits=4, method=method,
                                window_id=f'w{index}', cluster_id=article, kl=reference + difference,
                                baseline_nll=2., nll_delta=difference, flip_rate=0.))
    frame = pd.DataFrame(records)
    interval = article_intervals(frame[frame.method == 'g_geometric'], ('kl',))['kl']
    assert interval == pytest.approx((10 / 3, 2., 6.))
    paired = paired_comparisons(frame)
    assert len(paired) == 9
    for row in paired[paired.metric == 'kl'].itertuples():
        assert (row.mean, row.lower, row.upper) == pytest.approx((2., 1., 4.))
    incomplete = frame.drop(frame[(frame.method == 'uniform_mse') & (frame.window_id == 'w2')].index)
    with pytest.raises(ValueError, match='identical windows'):
        paired_comparisons(incomplete)
