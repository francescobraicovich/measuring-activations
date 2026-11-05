"""Batch-local causal evaluation of projection-input ablations."""

import torch
import torch.nn.functional as F

from .masks import build_mask, condition_seed, random_mask


def prediction_metrics(baseline_logits, intervention_logits, input_ids, start=16, stop=127):
    """Per-window KL, NLL change and top-1 flip at next-token positions [start, stop)."""
    if baseline_logits.shape != intervention_logits.shape or baseline_logits.ndim != 3:
        raise ValueError("logits must have matching [batch, sequence, vocabulary] shapes")
    if input_ids.shape != baseline_logits.shape[:2] or not 0 <= start < stop < input_ids.shape[1]:
        raise ValueError("invalid input shape or scoring positions")
    baseline = F.log_softmax(baseline_logits[:, start:stop].float(), dim=-1)
    intervention = F.log_softmax(intervention_logits[:, start:stop].float(), dim=-1)
    if not torch.isfinite(baseline).all() or not torch.isfinite(intervention).all():
        raise ValueError("non-finite log probabilities")
    targets = input_ids[:, start + 1:stop + 1].unsqueeze(-1)
    kl = (baseline.exp() * (baseline - intervention)).sum(-1).mean(-1)
    baseline_nll = -baseline.gather(-1, targets).squeeze(-1).mean(-1)
    changed_nll = -intervention.gather(-1, targets).squeeze(-1).mean(-1)
    flips = (baseline.argmax(-1) != intervention.argmax(-1)).float().mean(-1)
    return {"kl": kl, "baseline_nll": baseline_nll, "nll_delta": changed_nll - baseline_nll, "flip_rate": flips}


def evaluate_batch(model, projection, input_ids, windows, model_name, layer, method, fraction,
                   thresholds, outgoing_norms, repetition=0, random_control=False,
                   start=16, stop=127, baseline_logits=None):
    """Evaluate one condition and return per-window metrics and removed signal fractions.

    `thresholds` must be fitted on train. The caller is responsible for enforcing that
    provenance and the descriptive gate before invoking held-out evaluation.
    """
    if len(windows) != input_ids.shape[0]:
        raise ValueError("window count must match batch size")
    statistics = {}
    norms = torch.as_tensor(outgoing_norms, device=input_ids.device, dtype=torch.float32)

    def hook(_module, inputs):
        activations = inputs[0]
        mask = build_mask(activations, method, fraction, thresholds, norms)
        if random_control:
            counts = mask.sum(-1)
            mask = torch.stack([random_mask(mask[i].sum(-1), activations.shape[-1],
                condition_seed(model_name, layer, window.window_id, method, fraction, repetition))
                for i, window in enumerate(windows)])
            if not torch.equal(mask.sum(-1), counts):
                raise RuntimeError("random control does not match removal counts")
        absolute = activations.float().abs()
        weighted = absolute * norms
        axes = tuple(range(1, activations.ndim))
        statistics["observed_fraction"] = mask.float().mean(dim=axes)
        for name, values in (("removed_abs_fraction", absolute),
                             ("removed_squared_fraction", absolute.square()),
                             ("removed_weighted_fraction", weighted)):
            total = values.sum(dim=axes)
            removed = (values * mask).sum(dim=axes)
            statistics[name] = torch.where(total > 0, removed / total, torch.zeros_like(total))
        return (torch.where(mask, torch.zeros_like(activations), activations), *inputs[1:])

    with torch.inference_mode():
        baseline = baseline_logits if baseline_logits is not None else model(input_ids, use_cache=False).logits
        handle = projection.register_forward_pre_hook(hook)
        try:
            changed = model(input_ids, use_cache=False).logits
        finally:
            handle.remove()
        metrics = prediction_metrics(baseline, changed, input_ids, start, stop)
    return [{**{key: float(value[i].cpu()) for key, value in metrics.items()},
             **{key: float(value[i].cpu()) for key, value in statistics.items()}}
            for i in range(len(windows))]
