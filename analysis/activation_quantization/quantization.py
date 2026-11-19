"""Symmetric uniform and increasing-density SD-band codebooks."""
import numpy as np
import torch


CLIP_MULTIPLIERS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
METHODS = ('uniform_max', 'uniform_mse', 'g_geometric', 'uniform_matched')


def positive_levels(method, bits, sigma, clip):
    """Return zero and positive levels; mirror them to obtain 2**bits-1 values."""
    if method not in ('uniform', 'geometric'):
        raise ValueError('method must be uniform or geometric')
    if not isinstance(bits, int) or not 2 <= bits <= 16:
        raise ValueError('bits must be an integer between 2 and 16')
    if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(clip) or clip <= 0:
        raise ValueError('sigma and clip must be finite and positive')
    count = 2 ** (bits - 1) - 1
    if method == 'uniform':
        return np.linspace(0, clip, count + 1, dtype=np.float64)
    band_count = int(np.ceil(clip / sigma))
    boundaries = np.minimum(np.arange(band_count + 1, dtype=np.float64) * sigma, clip)
    # Subtracting the largest exponent avoids overflowing for large outliers.
    density = np.exp2(np.arange(band_count, dtype=np.float64) - (band_count - 1))
    cumulative = np.concatenate(([0.], np.cumsum(np.diff(boundaries) * density)))
    coordinates = np.linspace(0, cumulative[-1], count + 1)
    levels = np.interp(coordinates, cumulative, boundaries)
    levels[0], levels[-1] = 0., clip
    return levels


def reconstruct(values, levels):
    """Nearest symmetric level, with midpoint ties assigned toward zero."""
    values = np.asarray(values)
    levels = np.asarray(levels)
    boundaries = (levels[:-1] + levels[1:]) / 2
    indices = np.searchsorted(boundaries, np.abs(values), side='left')
    return np.sign(values) * levels[indices]


def fit_codebooks(sample, sigma, maximum, bits):
    """Choose clipping on training activation MSE using an identical range grid."""
    sample = np.asarray(sample, dtype=np.float64).reshape(-1)
    if not sample.size or not np.isfinite(sample).all():
        raise ValueError('calibration sample must be nonempty and finite')
    positive_levels('uniform', bits, sigma, maximum)
    clips = sorted({min(multiplier * sigma, maximum) for multiplier in CLIP_MULTIPLIERS} | {maximum})
    candidates = []
    selected = {}
    for kind in ('uniform', 'geometric'):
        records = []
        for clip in clips:
            levels = positive_levels(kind, bits, sigma, clip)
            mse = float(np.mean((reconstruct(sample, levels) - sample) ** 2))
            record = {'kind': kind, 'clip': clip, 'clip_sd': clip / sigma, 'mse': mse}
            candidates.append(record)
            records.append(record)
        selected[kind] = min(records, key=lambda record: record['mse'])
    configuration = {
        'uniform_max': ('uniform', maximum),
        'uniform_mse': ('uniform', selected['uniform']['clip']),
        'g_geometric': ('geometric', selected['geometric']['clip']),
        'uniform_matched': ('uniform', selected['geometric']['clip']),
    }
    result = {}
    for method, (kind, clip) in configuration.items():
        levels = positive_levels(kind, bits, sigma, clip)
        result[method] = {'levels': levels.tolist(), 'clip': clip, 'clip_sd': clip / sigma,
                          'level_count': 2 * len(levels) - 1,
                          'calibration_mse': float(np.mean((reconstruct(sample, levels) - sample) ** 2)),
                          'calibration_clip_fraction': float(np.mean(np.abs(sample) > clip))}
    return result, candidates


class ActivationQuantizer:
    """Apply a fixed codebook on the activation's device."""

    def __init__(self, levels, device='cpu'):
        levels = np.asarray(levels)
        if (levels.ndim != 1 or len(levels) < 2 or levels[0] != 0
                or not np.isfinite(levels).all() or not np.all(np.diff(levels) > 0)):
            raise ValueError('levels must start at zero and strictly increase')
        self.levels = torch.as_tensor(levels.copy(), dtype=torch.float32, device=device)
        self.boundaries = (self.levels[:-1] + self.levels[1:]) / 2

    def __call__(self, activation):
        absolute = activation.abs().contiguous()
        if activation.device.type == 'mps':
            # Vectorized binary search avoids the very slow MPS bucketize kernel.
            lower = torch.zeros_like(absolute, dtype=torch.int32)
            upper = torch.full_like(lower, len(self.boundaries))
            for _ in range(len(self.boundaries).bit_length()):
                middle = (lower + upper) // 2
                boundary = self.boundaries[middle.clamp(max=len(self.boundaries) - 1).long()]
                right = (lower < upper) & (absolute > boundary)
                lower = torch.where(right, middle + 1, lower)
                upper = torch.where(right, upper, middle)
            indices = lower.long()
        else:
            indices = torch.bucketize(absolute, self.boundaries, right=False)
        return activation.sign() * self.levels[indices]


def quantized_forward(model, modules, quantizers, ids):
    """Install codebooks for one forward pass and always remove the hooks."""
    if len(modules) != len(quantizers):
        raise ValueError('each projection must have one quantizer')
    handles = []
    for module, quantizer in zip(modules, quantizers):
        def intervene(_module, inputs, quantizer=quantizer):
            return (quantizer(inputs[0]), *inputs[1:])
        handles.append(module.register_forward_pre_hook(intervene))
    try:
        return model(ids, use_cache=False).logits
    finally:
        for handle in handles:
            handle.remove()
