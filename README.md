# Weak MLP activation ablation

[Read the article: What is a weak activation?](blog.html) It compares neuron-specific, pooled, within-token, and output-weighted definitions of weak activation in the context of the Cammarata and Nanda discussion. An interactive eight-channel example shows exactly how the four masks differ. Two static scientific figures and the full empirical explorer expose both favorable results and counterexamples. All interventions zero the selected MLP output-projection inputs; models and weights remain frozen.

The evaluation uses 512 nonoverlapping 128-token windows per model from the same 61 WikiText-2 articles. The report defaults to SmolLM2 and offers a GPT-2 toggle, layer and quantile controls, paired comparisons, and tables. Results are exploratory; tokenizer-specific spans differ and the random comparator uses one seed.

## Reproduce

Use Python 3.11 or 3.12, Apple Silicon with MPS or CPU, and the pinned dependencies. Node.js is required for the JavaScript/Python parity test. Model and dataset downloads require network access on the first run. The full sweep takes several hours; only one model should be loaded at a time.

```sh
uv venv --python 3.12
uv pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python analysis/large_study/evaluate.py validation --model smollm2 --layers 16 --count 4 --batch-size 8 --device mps
.venv/bin/python analysis/large_study/evaluate.py validation --model gpt2 --layers 6 --count 4 --batch-size 8 --device mps
.venv/bin/python analysis/large_study/evaluate.py test --model smollm2 --batch-size 8 --device mps
.venv/bin/python analysis/large_study/evaluate.py test --model gpt2 --batch-size 8 --device mps
.venv/bin/python analysis/large_study/visualize.py
.venv/bin/python analysis/large_study/write_report.py
```

Use `--device cpu` where MPS is unavailable; reduce batch size if necessary. If using the cache from the completed run, set `HF_HOME=/tmp/measuring-activations-huggingface` and enable `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`. The runner checks both-model validation and source/calibration hashes, checkpoints each batch, and resumes only with the same inputs and configuration. Training activations are collected locally when compatible saved calibration inputs are unavailable.

To rebuild only the article from the saved results, run the final two build commands above. Python and Node.js are required; no model inference or network access is needed. The editable snapshot [blog.md](blog.md) includes the article, figure captions, and explorer text. After editing it, ask for the changes to be incorporated into the post; rebuilding does not automatically import Markdown edits. The source prose is in `analysis/large_study/report_content.html`; `write_report.py` fills its numerical claims from saved tables and rejects missing or duplicate measurements. Mathematics uses LaTeX delimiters `\(...\)` and `\[...\]`, typeset at build time by `render_math.js` with the pinned KaTeX assets in `assets/vendor/katex/`. The generated article embeds its math fonts and renderer, so static equations and interactive notation work offline; static equations also retain accessible MathML and render with JavaScript disabled. `report_figures.py` exports SVG and PNG versions to `results/large_study/figures/`. The interactive definition calculation is in `strength_definitions.js` and is tested against the actual experimental mask implementation, including ties and endpoint fractions. Browser layout was not inspected during the rewrite because the available browser blocked local-file URLs; the calculations, generated markup, and exported figures were checked offline.

The protocol is in [analysis/large_study/PROTOCOL.md](analysis/large_study/PROTOCOL.md). The sample manifest and results are in `results/large_study/`; [per-window metrics](results/large_study/data/metrics.parquet), [paired comparisons](results/large_study/data/paired_comparisons.csv), and the [standalone explorer](results/large_study/interactive.html) are available. Per-layer outputs also contain calibration, run records, summaries, and checkpoint provenance.

## Interpretation

N and G use train-fitted strict-less-than thresholds, so their held-out event counts can differ from nominal fractions. L and W have exact fixed per-token quotas; the one shared random baseline matches those quotas, but not necessarily N/G. Reported 95% intervals resample articles, not tokens, and do not include variation from random-control seeds. KL measures deviation from the intact model in nats per scored token; it is not an accuracy measure or a proof that a neuron has been explained.
