#!/usr/bin/env python3
"""Run the preregistered study in order, reusing verified completed stages."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LAYERS = {"gpt2": (2, 6, 11), "smollm2": (5, 16, 29)}
TABLES = ("channels", "tokens", "removal", "token_removal", "overlap", "correlation", "density", "examples", "thresholds")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def stage(name, command, outputs, state_dir, inputs=()):
    """Reuse only when command, input hashes and every output hash still match."""
    marker = state_dir / f"{name}.json"
    signature = {"command": command, "inputs": {str(p): digest(p) for p in inputs}}
    if marker.exists():
        record = json.loads(marker.read_text())
        if record.get("signature") != signature or any(not Path(path).is_file() or digest(path) != checksum
                                                       for path, checksum in record.get("outputs", {}).items()):
            raise RuntimeError(f"completed stage {name} has changed; use a new output directory")
        if set(record.get("outputs", {})) != {str(path) for path in outputs}:
            raise RuntimeError(f"completed stage {name} has different outputs; use a new output directory")
        print(f"Reusing {name}", flush=True)
        return
    if any(path.exists() for path in outputs):
        raise RuntimeError(f"incomplete stage {name} has existing outputs; remove its output directory before retrying")
    print(f"Running {name}", flush=True)
    log = state_dir / f"{name}.log"
    with log.open("w") as stream:
        stream.write("Command: " + repr(command) + "\n")
        stream.flush()
        process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"stage {name} failed; see {log}")
    if any(not path.is_file() for path in outputs):
        raise RuntimeError(f"stage {name} did not produce all expected outputs; see {log}")
    write_json(marker, {"signature": signature, "outputs": {str(p): digest(p) for p in outputs}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "study")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--test-windows", type=int, choices=[64, 128], default=128)
    parser.add_argument("--stop-after-validation", action="store_true",
                        help="Stop after writing the validation gate, before held-out evaluation")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dataset-revision", default="main")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = output / "stages"
    state.mkdir(exist_ok=True)
    python = sys.executable
    sources = sorted([*ROOT.glob("src/*.py"), *ROOT.glob("scripts/*.py")])
    descriptions = []
    summaries = []
    figures = []
    for model, layers in LAYERS.items():
        for layer in layers:
            label = f"{model}_{layer}"
            directory = output / "descriptive" / label
            expected = [directory / f"{name}.csv" for name in TABLES] + [directory / "calibration.pt", directory / "metadata.json", directory / "threshold_ties.csv", directory / "activations.npy", directory / "outgoing_norms.npy", directory / "windows.csv"]
            command = [python, str(ROOT / "scripts/describe.py"), "--model", model, "--layer", str(layer), "--output", str(directory), "--batch-size", str(args.batch_size), "--device", args.device, "--model-revision", args.model_revision, "--dataset-revision", args.dataset_revision]
            stage(f"describe_{label}", command, expected, state, sources)
            figure_dir = output / "figures" / "descriptive" / label
            figure_outputs = [figure_dir / f"{name}.html" for name in ("signed_distributions", "magnitude_distributions", "channel_rms", "scale_weight", "weighted_scale", "jaccard", "rank_correlation", "density_N_L", "density_L_W", "removal", "token_removal", "scale_distributions", "token_concentration", "thresholds", "channel_removal", "channel_disagreement")]
            summary = figure_dir / "descriptive_summary.md"
            stage(f"figures_{label}", [python, str(ROOT / "scripts/make_figures.py"), str(directory), str(figure_dir)], figure_outputs + [summary], state, sources + expected)
            descriptions.append({"model": model, "layer": layer, "calibration": str(directory / "calibration.pt"), "sha256": digest(directory / "calibration.pt")})
            summaries.append(summary)
            figures.extend(figure_outputs)
    summary_path = output / "descriptive_summary.md"
    summary_path.write_text("\n\n".join(path.read_text() for path in summaries))
    descriptive_gate = output / "descriptive_complete.json"
    write_json(descriptive_gate, {"entries": descriptions, "figures": [str(path) for path in figures], "summary": str(summary_path)})
    validations = []
    for model, layer in (("gpt2", 6), ("smollm2", 16)):
        directory = output / "validation" / model
        calibration = output / "descriptive" / f"{model}_{layer}" / "calibration.pt"
        command = [python, str(ROOT / "scripts/ablate.py"), "--model", model, "--layer", str(layer), "--calibration", str(calibration), "--output", str(directory), "--split", "validation", "--windows", "4", "--batch-size", str(args.batch_size), "--device", args.device]
        expected = [directory / name for name in ("metrics.csv", "metrics.parquet", "paired_contrasts.csv", "run.json")]
        stage(f"validation_{model}", command, expected, state, sources + [calibration])
        validations.append({"model": model, "path": str(directory / "run.json")})
    validation_gate = output / "validation_complete.json"
    write_json(validation_gate, {"entries": validations})
    if args.stop_after_validation:
        print(f"Validation complete: {validation_gate}")
        return
    test_outputs = []
    for model, layers in LAYERS.items():
        for layer in layers:
            label = f"{model}_{layer}"
            directory = output / "test" / label
            calibration = output / "descriptive" / label / "calibration.pt"
            command = [python, str(ROOT / "scripts/ablate.py"), "--model", model, "--layer", str(layer), "--calibration", str(calibration), "--output", str(directory), "--split", "test", "--windows", str(args.test_windows), "--batch-size", str(args.batch_size), "--device", args.device, "--descriptive-gate", str(descriptive_gate), "--validation-gate", str(validation_gate)]
            expected = [directory / name for name in ("metrics.csv", "metrics.parquet", "paired_contrasts.csv", "run.json")]
            stage(f"test_{label}", command, expected, state, sources + [calibration, descriptive_gate, validation_gate] + [Path(v["path"]) for v in validations])
            test_outputs.append(directory)
    import pandas as pd
    metrics = pd.concat([pd.read_csv(path / "metrics.csv") for path in test_outputs], ignore_index=True)
    contrasts = pd.concat([pd.read_csv(path / "paired_contrasts.csv") for path in test_outputs], ignore_index=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    metrics.to_parquet(output / "metrics.parquet", index=False)
    contrasts.to_csv(output / "paired_contrasts.csv", index=False)
    causal = output / "figures" / "causal"
    stage("causal_figures", [python, str(ROOT / "scripts/make_figures.py"), str(output / "metrics.csv"), str(causal), "--causal", "--paired", str(output / "paired_contrasts.csv")], [causal / f"{name}.html" for name in ("causal_events", "causal_absolute_mass", "causal_weighted_mass", "depth", "paired_contrasts")], state, sources + [output / "metrics.csv", output / "paired_contrasts.csv"])
    print(f"Study complete: {output}")


if __name__ == "__main__":
    main()
