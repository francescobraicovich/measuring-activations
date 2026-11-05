"""Locate and record the input to the final MLP projection."""
import numpy as np
import torch


def projection(model, model_name, layer):
    if model_name in {"gpt2", "openai-community/gpt2"}:
        return model.transformer.h[layer].mlp.c_proj
    if model_name in {"smollm2", "HuggingFaceTB/SmolLM2-135M"}:
        return model.model.layers[layer].mlp.down_proj
    raise ValueError(f"unsupported model: {model_name}")


def outgoing_norms(model, model_name, layer):
    module = projection(model, model_name, layer)
    weight = module.weight.detach().float()
    dimension = 1 if model_name in {"gpt2", "openai-community/gpt2"} else 0
    norms = torch.linalg.vector_norm(weight, dim=dimension).cpu()
    expected = model.config.n_inner or 4 * model.config.n_embd if dimension == 1 else model.config.intermediate_size
    if norms.numel() != expected:
        raise ValueError(f"projection has {norms.numel()} channels, expected {expected}")
    return norms


def collect_activations(model, model_name, layer, windows, path, batch_size=2, device="cpu"):
    """Write unmodified projection inputs to a float32 CPU memmap [tokens, channels]."""
    if not windows or batch_size < 1:
        raise ValueError("windows must be nonempty and batch_size positive")
    size = len(windows[0].input_ids)
    if any(len(w.input_ids) != size for w in windows):
        raise ValueError("all windows must have equal length")
    channels = outgoing_norms(model, model_name, layer).numel()
    values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                       shape=(len(windows) * size, channels))
    offset = 0
    captures = []
    def capture(module, inputs):
        activation = inputs[0]
        if activation.ndim != 3 or activation.shape[-1] != channels:
            raise ValueError(f"unexpected projection input shape: {activation.shape}")
        captures.append(activation.detach().to(device="cpu", dtype=torch.float32).numpy())
    handle = projection(model, model_name, layer).register_forward_pre_hook(capture)
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(windows), batch_size):
                batch = windows[start:start + batch_size]
                ids = torch.tensor([w.input_ids for w in batch], dtype=torch.long, device=device)
                model(ids, use_cache=False)
                if len(captures) != 1 or captures[0].shape[:2] != (len(batch), size):
                    raise RuntimeError("projection hook did not capture exactly one batch")
                count = len(batch) * size
                values[offset:offset + count] = captures.pop().reshape(count, channels)
                offset += count
    finally:
        handle.remove()
        values.flush()
    return values
