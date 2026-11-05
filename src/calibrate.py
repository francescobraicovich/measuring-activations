"""Train-only percentile thresholds for historical activation magnitudes."""
import numpy as np
import torch


def fit_thresholds(activations, fractions, split="train"):
    if split != "train":
        raise ValueError("thresholds may only be fit on the train split")
    values = np.asarray(activations)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("activations must be a nonempty finite token-by-channel array")
    fractions = list(fractions)
    if any(not 0 <= p <= 1 for p in fractions):
        raise ValueError("fractions must be between zero and one")
    magnitude = np.abs(values)
    per_channel = np.quantile(magnitude, fractions, axis=0)
    pooled = np.quantile(magnitude, fractions)
    return {float(p): {"N": torch.as_tensor(per_channel[i], dtype=torch.float32),
                       "G": torch.as_tensor(pooled[i], dtype=torch.float32)}
            for i, p in enumerate(fractions)}


def threshold_tie_fractions(activations, thresholds):
    magnitude = np.abs(np.asarray(activations))
    return {p: {"N": np.mean(magnitude == values["N"].numpy(), axis=0),
                "G": float(np.mean(magnitude == values["G"].item()))}
            for p, values in thresholds.items()}


def save_calibration(path, thresholds, metadata):
    if metadata.get("split") != "train":
        raise ValueError("calibration provenance must identify the train split")
    torch.save({"thresholds": thresholds, "metadata": metadata}, path)


def load_calibration(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("metadata", {}).get("split") != "train":
        raise ValueError("calibration provenance does not identify the train split")
    return data
