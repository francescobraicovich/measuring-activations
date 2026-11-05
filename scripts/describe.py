#!/usr/bin/env python3
"""Collect train activations and produce descriptive tables for one layer."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from src.calibrate import fit_thresholds, save_calibration, threshold_tie_fractions
from src.data import load_windows
from src.distributions import describe_activations
from src.model_hooks import collect_activations, outgoing_norms

MODELS = {"gpt2": "openai-community/gpt2", "smollm2": "HuggingFaceTB/SmolLM2-135M"}
FRACTIONS = [0, .25, .5, .75, .9, 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dataset-revision", default="main")
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    args.output.mkdir(parents=True, exist_ok=True)
    model_id = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=args.model_revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=args.model_revision,
                                                 torch_dtype=torch.float32).to(args.device).eval()
    windows = load_windows(tokenizer, "train", args.windows, args.window_size, args.dataset_revision)
    pd.DataFrame([{"window_id": w.window_id, "passage_id": w.passage_id, "split": w.split}
                  for w in windows]).to_csv(args.output / "windows.csv", index=False)
    activations = collect_activations(model, args.model, args.layer, windows,
                                      args.output / "activations.npy", args.batch_size, args.device)
    norms = outgoing_norms(model, args.model, args.layer)
    np.save(args.output / "outgoing_norms.npy", norms.numpy())
    thresholds = fit_thresholds(activations, FRACTIONS, split="train")
    versions = {name: importlib.metadata.version(name) for name in
                ["torch", "transformers", "datasets", "numpy", "pandas", "scipy"]}
    metadata = {"split": "train", "model": model_id, "model_revision": args.model_revision,
                "model_commit": getattr(model.config, "_commit_hash", None),
                "tokenizer": tokenizer.name_or_path,
                "tokenizer_commit": tokenizer.init_kwargs.get("_commit_hash"),
                "dataset": "Salesforce/wikitext", "dataset_config": "wikitext-2-raw-v1",
                "dataset_revision": args.dataset_revision, "layer": args.layer,
                "window_count": len(windows), "window_size": args.window_size,
                "batch_size": args.batch_size, "device": args.device, "dtype": "float32",
                "fractions": FRACTIONS, "threshold_rule": "strictly less than; p=0 and p=1 overridden",
                "versions": versions}
    save_calibration(args.output / "calibration.pt", thresholds, metadata)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    ties = threshold_tie_fractions(activations, thresholds)
    pd.DataFrame([{"fraction": p, "channel": channel, "N": rate, "G": ties[p]["G"]}
                  for p in FRACTIONS for channel, rate in enumerate(ties[p]["N"])]).to_csv(
                      args.output / "threshold_ties.csv", index=False)
    tables = describe_activations(activations, norms, thresholds, FRACTIONS)
    for name, table in tables.items():
        table.to_csv(args.output / f"{name}.csv", index=False)
    print(f"Saved training descriptions for {model_id} layer {args.layer} in {args.output}")


if __name__ == "__main__":
    main()
