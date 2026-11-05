"""Activation-event masks used by the ablation protocol."""

import hashlib
import math

import torch


def build_mask(activations, method, fraction, thresholds=None, outgoing_norms=None):
    """Return a Boolean removal mask; calibrated thresholds use a strict < rule."""
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    if activations.ndim < 1 or activations.shape[-1] == 0:
        raise ValueError("activations must have a nonempty channel dimension")
    if method not in {"N", "L", "G", "W"}:
        raise ValueError(f"unknown method: {method}")
    if fraction == 0:
        return torch.zeros_like(activations, dtype=torch.bool)
    if fraction == 1:
        return torch.ones_like(activations, dtype=torch.bool)
    magnitude = activations.abs()
    if method in {"N", "G"}:
        if thresholds is None or method not in thresholds:
            raise ValueError(f"{method} requires calibration thresholds")
        threshold = torch.as_tensor(thresholds[method], device=activations.device, dtype=activations.dtype)
        if method == "N" and threshold.shape != (activations.shape[-1],):
            raise ValueError("N threshold must have one value per channel")
        if method == "G" and threshold.numel() != 1:
            raise ValueError("G threshold must be scalar")
        return magnitude < threshold
    scores = magnitude
    if method == "W":
        if outgoing_norms is None:
            raise ValueError("W requires outgoing norms")
        norms = torch.as_tensor(outgoing_norms, device=activations.device, dtype=activations.dtype)
        if norms.shape != (activations.shape[-1],):
            raise ValueError("outgoing norms must have one value per channel")
        scores = scores * norms
    count = math.floor(fraction * activations.shape[-1])
    # CPU stable sorting provides the same tie rule on backends without stable sort.
    order = torch.argsort(scores.detach().cpu(), dim=-1, stable=True).to(activations.device)
    mask = torch.zeros_like(activations, dtype=torch.bool)
    return mask.scatter(-1, order[..., :count], True)


def random_mask(counts, channels, seed):
    """Uniform independent channel samples without replacement for each token."""
    counts = torch.as_tensor(counts)
    if channels <= 0 or torch.any(counts < 0) or torch.any(counts > channels):
        raise ValueError("counts must be between zero and the channel count")
    if torch.any(counts != counts.floor()):
        raise ValueError("counts must be integers")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    flat = counts.detach().cpu().reshape(-1).to(torch.int64)
    result = torch.zeros((flat.numel(), channels), dtype=torch.bool)
    for index, count in enumerate(flat.tolist()):
        result[index, torch.randperm(channels, generator=generator)[:count]] = True
    return result.reshape(*counts.shape, channels).to(counts.device)


def condition_seed(model, layer, window_id, method, fraction, repetition):
    """Derive a stable seed independent of Python's randomized hash."""
    key = repr((model, layer, window_id, method, fraction, repetition)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63)
