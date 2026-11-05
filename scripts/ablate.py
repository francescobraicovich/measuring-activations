#!/usr/bin/env python3
"""Run validation or held-out ablations from train-only calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from src.calibrate import load_calibration
from src.data import load_windows
from src.evaluate import evaluate_batch
from src.model_hooks import outgoing_norms, projection

MODELS = {"gpt2": "openai-community/gpt2", "smollm2": "HuggingFaceTB/SmolLM2-135M"}
LAYERS = {"gpt2": {2: "early", 6: "middle", 11: "final"},
          "smollm2": {5: "early", 16: "middle", 29: "final"}}


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_summary(rows, replicates=1000, seed=0):
    """Bootstrap passage clusters for paired method-minus-random KL contrasts."""
    frame = pd.DataFrame(rows)
    results = []
    rng = np.random.default_rng(seed)
    for (model, layer, method, fraction), group in frame.groupby(["model", "layer", "method", "fraction"]):
        if method.endswith("_random") or method in {"identity", "all_zero"}:
            continue
        reference = frame[(frame.model == model) & (frame.layer == layer) &
                          (frame.method == method + "_random") & (frame.fraction == fraction)]
        if group.window_id.duplicated().any():
            raise ValueError("duplicate method window in paired comparison")
        if reference.duplicated(["window_id", "repetition"]).any():
            raise ValueError("duplicate random repetition in paired comparison")
        if set(group.window_id) != set(reference.window_id):
            raise ValueError("method and random comparison must contain the same windows")
        repetitions = reference.groupby("window_id").repetition.agg(lambda values: tuple(sorted(values)))
        if repetitions.nunique() != 1:
            raise ValueError("random windows must have matching repetitions")
        if not reference.groupby("window_id").cluster_id.nunique().eq(1).all():
            raise ValueError("random repetitions disagree on cluster")
        reference = reference.groupby("window_id", as_index=False).agg(
            random_kl=("kl", "mean"), random_cluster=("cluster_id", "first"))
        paired = group.merge(reference, on="window_id", validate="one_to_one")
        if not paired.cluster_id.eq(paired.random_cluster).all():
            raise ValueError("paired windows disagree on cluster")
        paired["contrast"] = paired.kl - paired.random_kl
        cluster_values = paired.groupby("cluster_id").contrast.agg(["sum", "count"])
        clusters = cluster_values.index
        # All windows have the same number of scored tokens. Weight sampled
        # passage clusters by their number of windows to preserve the token mean.
        estimates = []
        for _ in range(replicates):
            sampled = rng.integers(0, len(clusters), size=len(clusters))
            estimates.append(cluster_values["sum"].to_numpy()[sampled].sum() /
                             cluster_values["count"].to_numpy()[sampled].sum())
        results.append({"model": model, "layer": layer, "method": method, "fraction": fraction,
                        "paired_kl_difference": paired.contrast.mean(),
                        "lower": np.quantile(estimates, .025), "upper": np.quantile(estimates, .975),
                        "clusters": len(clusters), "replicates": replicates})
    return pd.DataFrame(results)


def validate_hook(model, module, ids, channels):
    captured = []
    def identity(_module, inputs):
        captured.append(inputs[0].shape)
        return inputs
    with torch.inference_mode():
        baseline = model(ids, use_cache=False).logits
        handle = module.register_forward_pre_hook(identity)
        try:
            hooked = model(ids, use_cache=False).logits
        finally:
            handle.remove()
    if captured != [torch.Size((*ids.shape, channels))] or not torch.allclose(baseline, hooked, rtol=1e-5, atol=1e-6):
        raise RuntimeError("identity hook or projection shape validation failed")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--descriptive-gate", type=Path, default=Path("results/descriptive_complete.json"))
    parser.add_argument("--validation-gate", type=Path, default=Path("results/validation_complete.json"))
    parser.add_argument("--windows", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    if args.layer not in LAYERS[args.model] or args.batch_size < 1:
        parser.error("layer must be prespecified and batch size positive")
    calibration = load_calibration(args.calibration)
    metadata = calibration["metadata"]
    if metadata["model"] != MODELS[args.model] or metadata["layer"] != args.layer or metadata["window_size"] != 128:
        raise ValueError("calibration model, layer or window size does not match")
    if args.split == "test":
        description = json.loads(args.descriptive_gate.read_text())
        expected = {(name, layer) for name, layers in LAYERS.items() for layer in layers}
        actual = {(entry["model"], entry["layer"]) for entry in description["entries"]}
        if actual != expected or not description.get("figures"):
            raise ValueError("descriptive gate must cover all six model/layer descriptions and figures")
        for entry in description["entries"]:
            path = Path(entry["calibration"])
            if file_digest(path) != entry["sha256"]:
                raise ValueError(f"descriptive calibration has changed: {path}")
            described = load_calibration(path)["metadata"]
            if described.get("model") != MODELS[entry["model"]] or described.get("layer") != entry["layer"]:
                raise ValueError(f"descriptive calibration does not match its entry: {path}")
            for name in ("channels", "tokens", "removal", "token_removal", "overlap", "correlation", "density", "examples"):
                if not (path.parent / f"{name}.csv").is_file():
                    raise ValueError(f"missing descriptive table: {path.parent / f'{name}.csv'}")
        current_entry = next(entry for entry in description["entries"]
                             if entry["model"] == args.model and entry["layer"] == args.layer)
        if Path(current_entry["calibration"]).resolve() != args.calibration.resolve():
            raise ValueError("test calibration must be the calibration used for the descriptive gate")
        for path in [description["summary"], *description["figures"]]:
            if not Path(path).is_file():
                raise ValueError(f"missing descriptive artifact: {path}")
        validation = json.loads(args.validation_gate.read_text())
        records = [json.loads(Path(entry["path"]).read_text()) for entry in validation["entries"]]
        if {record.get("model") for record in records} != set(MODELS):
            raise ValueError("validation gate must cover both models")
        if any(record.get("layer") != (6 if record.get("model") == "gpt2" else 16) for record in records):
            raise ValueError("validation sweeps must cover the middle layer")
        matching = next(record for record in records if record["model"] == args.model)
        for key in ("model_commit", "tokenizer_commit", "dataset_revision"):
            if matching["calibration"].get(key) != metadata.get(key):
                raise ValueError(f"validation and calibration disagree on {key}")
        if LAYERS[args.model][args.layer] == "middle" and matching.get("calibration_sha256") != file_digest(args.calibration):
            raise ValueError("validation and test calibration differ")
        if any(record.get("split") != "validation" or record.get("status") != "passed" or
               record.get("window_count", 0) < 4 for record in records):
            raise ValueError("validation gate contains an incomplete sweep")
    count = args.windows if args.windows is not None else (4 if args.split == "validation" else 128)
    if (args.split == "validation" and count < 4) or (args.split == "test" and count not in (64, 128)):
        raise ValueError("validation needs at least 4 windows; test requires 64 or 128")
    model_id = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=metadata["model_revision"])
    if tokenizer.init_kwargs.get("_commit_hash") != metadata.get("tokenizer_commit"):
        raise ValueError("tokenizer revision differs from calibration")
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=metadata["model_revision"], torch_dtype=torch.float32).to(args.device).eval()
    if getattr(model.config, "_commit_hash", None) != metadata.get("model_commit"):
        raise ValueError("model revision differs from calibration")
    windows = load_windows(tokenizer, args.split, count, 128, metadata["dataset_revision"])
    module = projection(model, args.model, args.layer)
    norms = outgoing_norms(model, args.model, args.layer).to(args.device)
    validate_hook(model, module, torch.tensor([windows[0].input_ids], device=args.device), len(norms))
    fractions = [.25, .5, .75, .9] if LAYERS[args.model][args.layer] == "middle" else [.75]
    rows = []
    for offset in range(0, len(windows), args.batch_size):
        batch = windows[offset:offset + args.batch_size]
        ids = torch.tensor([window.input_ids for window in batch], device=args.device)
        with torch.inference_mode():
            baseline = model(ids, use_cache=False).logits
        conditions = [(method, fraction, False, 0) for fraction in fractions for method in "NLGW"]
        conditions += [(method, fraction, True, repetition) for fraction in fractions for method in "NLGW"
                       for repetition in range(3 if fraction == .75 and LAYERS[args.model][args.layer] == "middle" else 1)]
        conditions += [("L", 0., False, 0), ("L", 1., False, 0)]
        for method, fraction, random_control, repetition in conditions:
            values = evaluate_batch(model, module, ids, batch, args.model, args.layer, method, fraction,
                                    calibration["thresholds"].get(fraction), norms, repetition,
                                    random_control, baseline_logits=baseline)
            for window, metrics in zip(batch, values):
                label = ("identity" if fraction == 0 else "all_zero" if fraction == 1 else
                         method + "_random" if random_control else method)
                rows.append(dict(model=args.model, layer=args.layer, depth=LAYERS[args.model][args.layer],
                                 window_id=window.window_id, cluster_id=window.passage_id, method=label,
                                 fraction=fraction, repetition=repetition, **metrics))
        if any(row["observed_fraction"] != 1 for row in rows if row["method"] == "all_zero"):
            raise RuntimeError("all-zero intervention did not remove all activations")
        if any(abs(row["kl"]) > 1e-6 for row in rows if row["method"] == "identity"):
            raise RuntimeError("identity intervention changed predictions")
    args.output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "metrics.csv", index=False)
    table.to_parquet(args.output / "metrics.parquet", index=False)
    paired_summary(rows).to_csv(args.output / "paired_contrasts.csv", index=False)
    record = dict(model=args.model, model_id=model_id, layer=args.layer, split=args.split,
                  status="passed", window_count=count, batch_size=args.batch_size, device=args.device,
                  calibration=metadata, calibration_sha256=file_digest(args.calibration),
                  descriptive_gate=str(args.descriptive_gate), seed_scheme="SHA-256 condition tuple")
    (args.output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved {len(rows)} per-window rows to {args.output}")


if __name__ == "__main__":
    main()
