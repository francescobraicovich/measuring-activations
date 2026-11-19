# Weak activation ablation

## Sample and intervention

The study uses frozen GPT-2 Small and SmolLM2-135M. After inspecting the pilot results, I sampled 512 nonoverlapping 128-token windows per model, cycling through 61 WikiText-2 test articles in source order. Both tokenizers sample the same articles but can cover different text. The [sample manifest](../../results/large_study/sample_manifest.json) records token offsets, input hashes and the dataset revision.

| Model | Layers tested individually | Checkpoint revision |
| --- | --- | --- |
| GPT-2 Small | 0, 2, 6, 8, 11 | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| SmolLM2-135M | 0, 5, 16, 21, 29 | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` |

Each intervention zeros selected MLP activations immediately before the output projection. It covers all 128 positions. Next-token outcomes are scored at positions 16 through 126.

| Rule | How it selects activations |
| --- | --- |
| N | Remove magnitudes below a training percentile fitted separately for each neuron. |
| G | Use one training percentile for all neurons in a layer. |
| L | At each token, remove the channels with the smallest magnitudes. |
| W | Rank channels at each token by magnitude times the norm of the outgoing weight vector. |

I also ranked activations by distance from each channel's signed training mean, then zeroed the original values. L and W remove a fixed count per token. N and G remove a variable count on held-out text. The random control uses the same per-token quota as L and W, taking progressively longer prefixes of one seeded channel permutation per token.

The removal grid is 10, 20, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 99, 99.5, 99.9 and 100 percent. The intact model supplies the baseline. Outcomes include KL divergence in nats per scored token, change in negative log likelihood and top-token flip rate. The 95% intervals resample whole articles 1,000 times with their windows kept together. They cover sampling variation for one random seed.

## All layers

The sequential run uses the first 64 windows. Each layer selects activations after the preceding layers have been intervened on. Calibration uses intact training activations, with 128 training windows for previously uncalibrated layers. The reported removal and removed-signal fractions average the layer-level fractions. The published tables combine two sweeps over disjoint interior fractions after checking their windows, calibration, code and endpoints.

## Rebuild the article

From the repository root, use Python 3.12 and Node.js.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python article/source/build.py
python -m pytest -q
```

Edit the prose in [the article template](../../article/source/article.html). The build fills in the measurements and writes `article/index.html`.

## Run the experiments

The first run downloads the models and WikiText-2. The full sweep takes several hours. Use `device=mps` on Apple Silicon or `device=cpu` for CPU execution.

```sh
device=mps
study=analysis/large_study
python "$study/evaluate.py" validation --model gpt2 --layers 6 --count 4 --batch-size 8 --device "$device"
python "$study/evaluate.py" validation --model smollm2 --layers 16 --count 4 --batch-size 8 --device "$device"

for model in gpt2 smollm2; do
  python "$study/evaluate.py" test --model "$model" --batch-size 8 --device "$device"
  python "$study/all_layers.py" --model "$model" --batch-size 8 --device "$device" \
    --fractions .1 .2 .3 .4 .5 .6 .7 .75 .8 .85 .9 .95 .99 .995 .999 \
    --output results/large_study/reproduction
done
```

Both validation runs must pass before test evaluation. Single-layer runs write to `results/large_study/test/<model>_<layer>/`. The command above collects the full all-layer grid in one sweep and writes to `results/large_study/reproduction/<model>/`. Compare each run's `summary.csv` and `metrics.parquet` with the [single-layer](../../results/large_study/data/) and [all-layer tables](../../results/large_study/all_layers_combined/). Interrupted runs resume with the same configuration.
