#!/usr/bin/env python3
"""Check the real model hook sites, projection norms, and validation throughput."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data import load_windows
from src.model_hooks import projection, outgoing_norms
from scripts.ablate import validate_hook

MODELS = {"gpt2": ("openai-community/gpt2", 6, 3072),
          "smollm2": ("HuggingFaceTB/SmolLM2-135M", 16, 1536)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    model_id, layer, channels = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(args.device).eval()
    windows = load_windows(tokenizer, "validation", 4, 128, "main")
    module = projection(model, args.model, layer)
    norms = outgoing_norms(model, args.model, layer)
    assert len(norms) == channels
    ids = torch.tensor([window.input_ids for window in windows[:2]], device=args.device)
    start = time.perf_counter()
    validate_hook(model, module, ids, channels)
    with torch.inference_mode():
        zero = torch.zeros((1, 1, channels), device=args.device)
        single = zero.clone()
        single[..., 0] = 1
        difference = module(single) - module(zero)
        assert torch.allclose(difference.norm().cpu(), norms[0], rtol=1e-4, atol=1e-6)
        handle = module.register_forward_pre_hook(lambda _module, inputs: (torch.zeros_like(inputs[0]), *inputs[1:]))
        try:
            logits = model(ids, use_cache=False).logits
        finally:
            handle.remove()
        assert torch.isfinite(logits).all()
    if args.device == "mps":
        torch.mps.synchronize()
    print(f"{model_id}: layer {layer}, channels {channels}, identity and one-channel projection passed, elapsed {time.perf_counter()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
