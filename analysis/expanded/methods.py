"""Train-calibrated original and mean-centered activation-event masks."""

import hashlib

import numpy as np
import torch

from src.calibrate import fit_thresholds, threshold_tie_fractions
from src.masks import build_mask as original_mask
from src.masks import condition_seed, random_mask


FRACTIONS = (0., .1, .2, .25, .3, .4, .5, .6, .7, .75, .8, .9, .95, .99, 1.)
METHODS = ("N", "L", "G", "W", "N_centered", "L_centered", "G_centered", "W_centered")


def fit_calibration(activations, fractions=FRACTIONS, split="train"):
    """Fit training means and thresholds, and report equality with each threshold.

    Centering is used only for scoring; removed activations are still set to zero.
    """
    if split != "train":
        raise ValueError("calibration may only be fit on the train split")
    values = np.asarray(activations)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("activations must be a nonempty finite token-by-channel array")
    fractions = list(fractions)
    means = torch.as_tensor(values.mean(axis=0), dtype=torch.float32)
    centered = values - means.numpy()
    original = fit_thresholds(values, fractions, split=split)
    centered_thresholds = fit_thresholds(centered, fractions, split=split)
    original_ties = threshold_tie_fractions(values, original)
    centered_ties = threshold_tie_fractions(centered, centered_thresholds)
    thresholds = {}
    ties = {}
    for p in original:
        thresholds[p] = {**original[p], **{f"{key}_centered": value for key, value in centered_thresholds[p].items()}}
        ties[p] = {**original_ties[p], **{f"{key}_centered": value for key, value in centered_ties[p].items()}}
    return {"means": means, "thresholds": thresholds, "tie_fractions": ties,
            "training_activations_sha256": hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
            "means_sha256": hashlib.sha256(means.numpy().tobytes()).hexdigest()}


def build_mask(activations, method, fraction, thresholds=None, outgoing_norms=None, means=None):
    """Return a removal mask using original or training-mean-centered scores."""
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    if method.endswith("_centered"):
        if means is None:
            raise ValueError("centered methods require training means")
        means = torch.as_tensor(means, device=activations.device, dtype=activations.dtype)
        if activations.ndim < 1 or means.shape != (activations.shape[-1],):
            raise ValueError("means must have one value per channel")
        activations = activations - means
        original_method = method.removesuffix("_centered")
        if thresholds is not None and method in thresholds:
            thresholds = {original_method: thresholds[method]}
        method = original_method
    return original_mask(activations, method, fraction, thresholds, outgoing_norms)
