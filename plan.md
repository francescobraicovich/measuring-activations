# Which weak MLP activations matter?

**Study protocol — weak-activation ablation only (Experiment A)**  
**Status:** planned; no empirical results yet  
**Hardware:** Apple Silicon Mac, target 16 GB unified memory; fully local after downloading models and dataset  
**Training:** none. Both models are publicly available pretrained *base* checkpoints and remain frozen.

## 1. Research question

**Which definition of a “weak” MLP activation best identifies activation events that can be zeroed without changing a language model's predictions?**

Evaluate four *percentile-based rankings*, always using the same intervention: zero selected post-nonlinearity MLP **channel activations at selected token positions**. We do not permanently remove neurons or modify weights.

1. **N — per-neuron percentile:** small relative to this neuron's own historical activation distribution.
2. **L — within-token layer percentile:** small relative to other channels in this layer **on the current token**.
3. **G — global layer magnitude percentile:** small relative to the pooled historical distribution of all activation events in this layer.
4. **W — output-weighted percentile:** small relative to other channels **on the current token**, after weighting each activation magnitude by its outgoing weight-vector norm.

The key distinction is *ranking*, not just threshold choice. N uses a neuron-specific calibrated reference; G uses a single calibrated layer-wide reference; L and W use token-wise relative rankings. All four must be studied descriptively **before** causal ablations.

### Questions and hypotheses

- **RQ1 (distributions):** How different are neuron activation scales, tails, signs, and outgoing weight norms? Are typical activation scale and output-weight norm systematically related?
- **RQ2 (rankings):** How often do N, L, G, and W label different activation events “weak,” and which kinds of neurons cause disagreement?
- **RQ3 (causal):** At comparable numbers of removed events, which ranking produces the lowest downstream KL compared with the intact model? Does each outperform count-matched random removal?
- **RQ4 (robustness):** Do results replicate across an older and a newer pretrained model, and at early/middle/late depths?

**Working hypotheses, not assumptions:** magnitude-ranked removals outperform random removal; the four rankings disagree because neuron scales differ; outgoing-weight norms help explain some disagreement; the advantage of any method may shrink when comparisons are matched by total removed signal.

## 2. Models, layers, and local execution

| Property | GPT-2 Small | SmolLM2-135M |
|---|---|---|
| Hugging Face ID | `openai-community/gpt2` | `HuggingFaceTB/SmolLM2-135M` |
| Release family | 2019 | 2024 |
| Parameters | ~124M | ~135M |
| Transformer layers | 12 | 30 |
| Hidden width | 768 | 576 |
| MLP channels | 3,072 | 1,536 |
| MLP computation | GELU followed by output projection | SwiGLU (`SiLU(gate) * up`) followed by down projection |
| Modern components | GPT-2-era architecture | RoPE, RMSNorm, grouped-query attention, SwiGLU |
| Checkpoint | Base pretrained | **Base pretrained, not Instruct** |

SmolLM2 is much newer and more data-trained at similar parameter count; it is a plausible stronger model, but **do not assume** it performs better on our particular corpus. Record each model's unmodified negative log-likelihood and, optionally, bits per byte on comparable raw text. Token-level perplexities across different tokenizers are *not* directly comparable.

**Intervene at the input to the final MLP projection:**

- GPT-2: `model.transformer.h[l].mlp.c_proj` receives the **post-GELU** activation vector.
- SmolLM2: `model.model.layers[l].mlp.down_proj` receives **post-SwiGLU-product** activation vector, `SiLU(gate_proj(x)) * up_proj(x)`.

Both are vectors `a[t,j]` followed by a linear output projection, so the local contribution of channel `j` is exactly `a[t,j] * w_out[j]` (apart from any unchanged projection bias). **Do not hook SmolLM2's `gate_proj` or the SiLU output alone**: that would study a different quantity.

The models have different depths. Compare approximately corresponding relative depths, `l/(n_layers-1)`:

| Depth | GPT-2 index | SmolLM2 index | Coverage |
|---|---:|---:|---|
| Early | 2 | 5 | Distributions + reduced 75% ablation |
| Middle, primary | 6 | 16 | All distribution analyses + full ablation sweep |
| Final | 11 | 29 | Distributions + reduced 75% ablation |

**Mac target:** one model loaded at a time, `eval()`, `torch.inference_mode()`, `float32`, `use_cache=False`, batch size 2 (decrease to 1 if needed), 128-token windows. Prefer `mps` after smoke testing; use CPU if needed. No cloud inference, training, or fine-tuning. The 30-layer model may take longer despite similar parameter count: benchmark on validation before choosing test batch size.

## 3. Exact definitions of the four percentile methods

For an MLP layer with `D` channels, let `a[t,j]` be the post-activation value for token `t`, channel `j`. Define `m[t,j] = abs(a[t,j])`. Let `v[j] = ||w_out[j]||_2` and `c[t,j] = m[t,j] * v[j]`. A **removal fraction** `p` always means removing the bottom `p` proportion of the *specified score distribution*, not removing entire neuron identities.

| Code | Score / reference population | Mask construction on new text | Does it use calibration? |
|---|---|---|---|
| **N** | Percentile of `m[t,j]` in the historical distribution of **channel j** | `m[t,j] < theta_N[j,p]`, with deterministic tie rule | Yes, per-channel thresholds |
| **L** | Percentile of `m[t,j]` across `j=1..D` at the **same token t** | Zero exactly `floor(p*D)` lowest `m[t,:]` values | No |
| **G** | Percentile of `m[t,j]` in **all channels and all calibration tokens** of this layer | `m[t,j] < theta_G[p]`, one common layer threshold | Yes, one pooled threshold |
| **W** | Percentile of `c[t,j] = m[t,j]*v[j]` across channels at the **same token t** | Zero exactly `floor(p*D)` lowest `c[t,:]` values | Uses frozen model output weights, no text calibration |

Compute `theta_N[j,p]` and `theta_G[p]` using **only calibration activations**. For L and W, rank **within each token**; W is explicitly an **output-weighted percentile**, *not* raw contribution magnitude as an absolute cutoff and *not* a calibrated global output-weighted threshold. W and L therefore remove exactly the same number of events at every token and form the cleanest head-to-head comparison of ranking quality.

All measures use **absolute values**, because GELU/SwiGLU channels can be negative. Analyze signs separately. All four use an identical **zero replacement** and leave the outgoing projection unchanged.

- `p in {0, .25, .50, .75, .90, 1}`. Set `p=0` to exact identity and `p=1` to exact all-zero, regardless of ties.
- Break L/W ties deterministically by channel index using a stable sort (or a tested CPU fallback). Verify exact removal counts.
- For N/G, use a documented threshold-side rule (`<` versus `<=`) and report actual held-out removal rates; ties or distribution shift may prevent exact removal of `p` of test events. Do **not** calculate calibration thresholds on held-out test text.
- For numerical zeros or degenerate per-neuron distributions, report the fraction tied at each threshold. Never silently discard those channels.

### Why these four are not redundant

- **N versus G:** asks whether normalizing by a channel's own history matters.
- **L versus G:** both order channels by raw magnitude *within a token*, but L fixes the number removed **per token**, whereas G applies a fixed calibration threshold and permits token-to-token variation in removal count. Any difference must be interpreted accordingly.
- **L versus W:** tests the effect of multiplying by the output-projection norm, holding the per-token removal budget *exactly fixed*.
- **N versus L/W:** tests historical unusualness versus instantaneous rank. Their realized held-out removal rates may differ.

**Important caveat:** L minimizes removed `sum |a|` among masks with the same per-token count; W analogously minimizes removed `sum |a_j|*||w_j||`. They optimize these **local proxies by construction**. A lower downstream KL would be new empirical evidence; a lower removed-proxy mass would **not** be independent evidence of causal superiority. Likewise, direct contribution norms ignore cancellation among channels and downstream nonlinearities.

## 4. Data and preregistration

Use `Salesforce/wikitext`, `wikitext-2-raw-v1`, retaining article/passage boundaries as far as the dataset permits. Use the same source passages for both model families where feasible, tokenized separately. Create deterministic, nonoverlapping, length-128 token windows **within each dataset split**.

| Partition | Windows per model | Purpose |
|---|---:|---|
| Train | 128 | Calibrate N and G; study distributions; select illustrative neurons |
| Validation | 32 | Smoke-test hooks, masks, runtime, threshold behavior |
| Test | 128 | Final causal ablation effects and confidence intervals |

Score next-token predictions at positions 16–126 of each window. Preserve passage/article IDs to support clustered uncertainty estimates; log tokenizer/revision. No test activations influence threshold fitting or the choice of explanatory examples. All analysis definitions, layer choices, `p` values, and figure templates should be frozen after the calibration and validation phases, *before* viewing test KL.

### Baseline reference

On the same evaluation text, record each model's intact NLL and top-token predictions. If comparing intrinsic model strength, use a separately reported tokenizer-normalized measure (e.g., bits per byte over aligned raw-text spans); do not compare per-token perplexities as though tokenizer units matched. This is contextual metadata, not a performance objective of this ablation study.

## 5. Phase 0 — descriptive distribution analysis **before causal intervention**

Run each model *without modifying activations*. Collect the target layer's `a[t,:]` to CPU in chunks; for full calibration, 128 × 128 × 3072 × 4 bytes is ~201 MB for GPT-2's MLP, versus ~101 MB for SmolLM2. Process one layer/model at a time; use NumPy memmaps or chunked storage rather than retaining all layers on MPS.

### 5.1 Neuron distributions and heterogeneity

For each channel `j` calculate, from calibration tokens:

- `mean(a_j)`, `median(a_j)`, `std(a_j)`, `RMS(a_j)`, `median(|a_j|)`, and `quantile(|a_j|, .25/.50/.75/.90/.99/.999)`;
- fraction `a_j < 0`, fraction effectively zero (with a stated numerical tolerance), positive and negative tail sizes, and `q99(|a_j|)/max(q50(|a_j|), eps)` as a tail-heaviness measure;
- `v_j = ||w_out[j]||_2` and typical **output-weighted scale** `RMS(a_j)*v_j`;
- rank stability of per-neuron scale: compare per-neuron medians/RMS across two nonoverlapping calibration halves to see if estimates are stable.

Show representative histograms of `a_j` (signed) and `log10(|a_j|+eps)` (magnitude), choosing channels deterministically from calibration quantiles of RMS, tail-heaviness, and output norm. Report the distribution of neuron-level statistics—not only a few cherry-picked examples. Compare GPT-2 GELU with SmolLM2 SwiGLU-product distributions without asserting that their channel semantics are identical.

### 5.2 Are large-activation neurons assigned small outgoing weights?

For each channel, plot

`r_j = sqrt(mean_cal a_j**2)` versus `v_j = ||w_out[j]||_2`,

on **log–log axes**, one panel per model; color by `fraction(a_j > 0)` or another preregistered activation statistic. Overlay binned median trends and report Spearman rank correlation; Pearson correlation on `log(r_j+eps), log(v_j+eps)` is secondary. Use bootstrap over *channels* for descriptive uncertainty, not token-level independence assumptions.

Additional panels/metrics:

1. Distribution of `r_j`, `v_j`, and `r_j*v_j` (compare dispersion and upper tails).
2. Compare the highest-RMS channels with the highest typical output-contribution channels: rank correlation and top-decile overlap.
3. Examine whether channels with small `r_j` nevertheless have large `v_j` (and vice versa), and whether this explains L/W rank disagreements.
4. Repeat scale–weight correlations at early/middle/final layers; avoid presenting three layers as a universal depth trend.
5. **Optional, cheap:** per-token cancellation ratio `sum_j |a_j|v_j / (||sum_j a_j*w_j||_2 + eps)` as context for why summed individual norms need not predict the net MLP output. Do not use this as an alternative strength ranking.

**First-principles motivation:** raw magnitude changes if one scales `a_j` by `c` and `w_out[j]` by `1/c` without changing their product, whereas `|a_j|*||w_out[j]||` is invariant under that coordinated reparameterization. This supports W as a geometrically meaningful *local* strength score, not a proof of causal importance downstream.

### 5.3 Token-level layer distributions and activity concentration

For each calibration token, compute the layer-wide median/p75/p90/p99 of `|a[t,:]|`, total `sum |a|`, and fraction of magnitude and squared mass in the top 10% of channels. Plot the distribution of per-token p75 cutoffs for L, versus the *single* G p75 threshold and the distribution of N p75 thresholds. Also plot how many channels G and N would remove per token: their variation is mechanistically relevant to L/W's fixed token-wise budgets.

### 5.4 How the four percentile rankings relate

For each calibration event `(t,j)`, calculate four **percentile scores**:

- `u_N[t,j] = empirical CDF_j(|a[t,j]|)` from channel history;
- `u_L[t,j] = percentile rank_j(|a[t,j]|)` within this token;
- `u_G[t,j] = empirical CDF_layer(|a[t,j]|)` from pooled layer history;
- `u_W[t,j] = percentile rank_j(|a[t,j]|*v_j)` within this token.

At `p=.75` (and optionally across all prespecified `p`), record:

- Pairwise **4×4 mask Jaccard overlap** and disagreement fraction, plus a paired 4×4 score-rank Spearman matrix (downsample events only for plotting if needed, not for summary metrics).
- Two informative joint-rank density plots: `N` vs `L` (historical unusualness vs within-token rank) and `L` vs `W` (effect of outgoing weights).
- Per-neuron rate of removal under each method, grouped by neuron RMS and outgoing norm; identify channels frequently kept by N but removed by L, and channels kept by W but removed by L.
- For each method, **actual removed event fraction** (global and per-token distribution), `sum_removed |a| / sum_all |a|`, `sum_removed a**2 / sum_all a**2`, and `sum_removed |a|v / sum_all |a|v`. Report these both at the nominal p75 cutoff and over the full p sweep.

**Interpretive guardrail:** G and L induce the same raw-magnitude *ordering within each token*, though their masks can differ because G removes a varying number of events per token. Do not treat disagreement between G and L as discovery of a different within-token neuron ordering.

**Phase-0 gate:** produce all distribution and relationship figures and a one-page descriptive summary before computing held-out causal metrics. Any changes to hypotheses after examining calibration must be labeled exploratory and not retroactively presented as preregistered.

## 6. Phase 1 — clean causal ablations

### Conditions

| Layer | Four methods | Random control | References |
|---|---|---|---|
| Primary middle | `p=.25,.50,.75,.90` for N/L/G/W | Event-count matched to each method and token | Identity, all-zero |
| Early/final | `p=.75` for N/L/G/W | Same count matching | Identity, all-zero |

For L and W, random controls can share the same per-token count because both remove `floor(p*D)` channels. N/G require independent per-token count-matched random masks if their observed counts differ. Repeat main-layer p75 random controls with three fixed seeds; use one seed at other settings. Every forward pass intervenes on **one selected layer only**.

**Exact control:** choose random channels *without replacement* independently for each token, using the number of events removed by that method on the same token. Seed reproducibly using `(model, layer, window_id, method, p, repetition)`; compare on identical inputs. Random masks match counts but not activation magnitude or per-channel firing frequency. No intervention selects events by looking at target next-token labels or downstream model outputs.

### Primary outcome

For a scored next-token position, let `P0` be the intact model distribution and `PT` the distribution after intervention:

`KL_t = sum_vocab P0(v) * (log P0(v) - log PT(v))`.

Report mean `KL(P0 || PT)` in **nats per scored token**, clustered intervals, and the paired contrast with each method's count-matched random baseline. Compute `log_softmax` in float32, verify finite outputs. Process one batch at a time: baseline logits → each intervention → aggregate per-window metrics; don't cache full-vocabulary logits across the dataset.

**Secondary metrics:** next-token NLL change, top-1 prediction flip rate, actual removal count, removed absolute/squared/output-weighted mass, and (optionally) change in local MLP output norm. Normalize KL by the all-zero layer reference *only when that denominator is meaningfully nonzero*, and label the ratio as a reference scale, not “percent of behavior preserved.”

### Prespecified comparisons

1. **Each method vs random at matched removed event counts.** This establishes whether its strength ranking adds information beyond removing the same number of events.
2. **L vs W** at equal *exact per-token counts*. This isolates whether outgoing weights improve selection relative to raw magnitude.
3. **N vs G** to test calibration per channel versus calibration per layer; show observed event counts and token-level removal variation.
4. **N/G vs L/W** on actual removed-fraction axes; never assert nominal p75 creates exactly equal test budgets. If counts differ, compare entire curves and report differences in realized fractions rather than claiming a strict equal-count superiority.
5. **Explain the tradeoff:** replot downstream KL against removed `sum|a|` and removed `sum|a|*v`. These are diagnostic reparameterizations; no method's advantage on the proxy it minimizes is independent evidence.
6. **Depth check:** compare p75 results at the three prespecified relative depths, with raw KL and all-zero reference side by side.

### Uncertainty and inference

Use **paired** bootstrap resampling over source documents/passages (1,000 replicates) to preserve matching across intervention conditions. If too few distinct articles exist, use passage/window clusters and disclose that the interval can be optimistic due to residual within-article dependence. Report raw paired contrasts with intervals rather than only statements that one curve is higher. No significance claims based on treating thousands of adjacent token positions as independent observations.

### Validation gates

- Identity pre-hook matches the no-hook model logits within float32 tolerance; all-zero correctly zeros the projection **input** and the layer usually has nontrivial impact.
- Verify module path, `[batch, 128, D]` shape (`D=3072` or `1536`), and `||w_out[j]||` against a one-channel projection unit test.
- Verify p0 is identity and p1 is all-zero; N/G threshold tensors come only from the calibration split.
- Verify L/W exact per-token counts and deterministic ties; random masks match each method's **actual** per-token counts.
- Run 4-window validation sweeps for **both** models before any full test sweep. Confirm that hook handles are removed after every run; no condition leaks into another.
- Save seed, model revision/hash, tokenizer, dataset revision, package versions, computation device/dtype, layer index, and config with results.

## 7. Figures and outputs

All figures are generated from saved numerical results (CSV/Parquet), with no fabricated data. Use consistent method colors and matching scales where sensible; export SVG and PNG.

1. **Distributions are heterogeneous.** Representative signed and absolute activation distributions plus distribution of channel RMS, for both models (middle layers).
2. **Does activation scale compensate for output weights?** Log–log scatter of channel RMS versus outgoing weight norm; Spearman correlation and binned trend, plus small panels for `RMS(a)*||w||`.
3. **Four rankings, four notions of weak.** Pairwise mask Jaccard and rank-correlation matrices (two models), with `N–L` and `L–W` joint-percentile density panels.
4. **What do the methods actually discard?** For all four, removal-rate curves and removed `|a|`, `a²`, and output-weighted mass; include variability of per-token N/G removal counts.
5. **The causal result.** Downstream KL versus **observed fraction of activation events removed** for N, L, G, W and count-matched random; one main-layer panel per model, paired uncertainty.
6. **Is the difference explained by signal magnitude?** Downstream KL versus removed raw activation mass, with a companion plot versus removed output-weighted mass.
7. **Depth check.** At p75, four methods plus matched random and full-zero baseline across corresponding early/middle/final layers.

**Minimum publishable artifact:** Figures 1–6, a reproducible `results/metrics.parquet`, `README.md`, and a concise report explaining both supportive and unsupportive outcomes. Figure 7 is the first optional element if the 30-layer model is slower than anticipated.

## 8. Implementation specification

**Dependencies:** Python 3.11+, `torch`, `transformers`, `datasets`, `numpy`, `pandas`, `scipy`, `matplotlib`, `pyarrow` (optional). No training framework, SAE, TransformerLens, paid API, or GPU server required.

Use ordinary Hugging Face `AutoModelForCausalLM` and a `register_forward_pre_hook` on the MLP's **output projection**. This obtains a single, architecture-independent intervention site despite different MLP nonlinearities. Validate module names with the installed Transformers version.

```python
# Projection modules; check these against loaded model objects.
gpt2_proj = model.transformer.h[layer].mlp.c_proj
smollm2_proj = model.model.layers[layer].mlp.down_proj

# Both receive post-nonlinearity channel vectors: [batch, seq, D].
def pre_hook(module, inputs):
    a = inputs[0]
    mask = build_mask(a, method, p, thresholds, outgoing_norms)
    a_new = torch.where(mask, torch.zeros_like(a), a)
    return (a_new, *inputs[1:])

handle = projection.register_forward_pre_hook(pre_hook)
try:
    with torch.inference_mode():
        logits = model(input_ids, use_cache=False).logits
finally:
    handle.remove()
```

**Projection weights:** HF GPT-2's `Conv1D` `c_proj.weight` is laid out `[D, d_model]`, so `v = norm(weight, dim=1)`. SmolLM2's `nn.Linear` `down_proj.weight` is `[d_model, D]`, so `v = norm(weight, dim=0)`. Assert both yield length `D`; verify numerical contribution with a sparse single-channel change. Compute `v` once from the frozen weight tensor in float32.

**Mask implementations:**

- `N`: compare `abs(a)` with `[1,1,D]` calibrated percentile thresholds.
- `G`: compare `abs(a)` with one calibrated layer threshold.
- `L`: rank `abs(a)` within the last dimension and mask bottom `floor(p*D)`.
- `W`: rank `abs(a) * v.reshape(1,1,D)` within the last dimension and mask bottom `floor(p*D)`.
- `R`: generate reproducible per-token uniformly random channel orders, mask the first `mask.sum(-1)` indices; assert count equality.

For L/W, test whether `torch.argsort(..., stable=True)` works on the chosen MPS/PyTorch combination; if not, fall back to a deterministic CPU sort/mask transfer rather than using undocumented or nondeterministic tie behavior. Keep MPS copies of calibrated thresholds and `v` during evaluation. Avoid in-place mutation of shared activation tensors.

### Suggested repository

```text
plan.md
requirements.txt
config.yaml
src/
  data.py             # passage-aware windows; per-model tokenization
  model_hooks.py      # locate output projections; identity/zero tests
  calibrate.py        # N/G thresholds, weight norms, activation summaries
  distributions.py    # RMS-vs-weight, score ranks, overlap and mass
  masks.py            # N/L/G/W masks and count-matched random masks
  evaluate.py         # local batched KL, NLL, flip metrics
  plots.py            # figures generated entirely from stored results
scripts/
  smoke_test.py
  describe.py         # MUST precede held-out causal evaluation
  ablate.py
  make_figures.py
results/
figures/
```

### Execution order and strict scope

1. Download both **pretrained base** checkpoints and WikiText-2; confirm both hooks and output-weight vectors.
2. Calibrate N/G from train, collect channel distributions and `v`, finish Figures 1–4 and the descriptive summary **before** test interventions.
3. Run validation-only identity/all-zero/mask tests and measure throughput; freeze the condition grid.
4. Run all layer-6 ablations on the **same held-out windows** for both models. Save per-window metrics, masks' aggregate counts/masses, and paired results.
5. Run reduced early/final-layer p75 checks; build Figures 5–7 and a 2–4 page write-up.

**Resource fallback:** reduce batch size first, then test windows from 128 to 64 **equally for both models and all four methods** if required; do not remove a particular ranking or model based on observed results. Keep calibration fixed. Record any reduction. No model training, no extra models, no quantization experiment, no semantic labels, no multi-layer simultaneous ablations.

## 9. Interpretation and limitations

- If **W outperforms L**, outgoing weights help rank *locally small* contributions under an exactly equal per-token ablation budget. This is not proof of downstream causal optimality.
- If **N outperforms G**, what is unusual for an individual channel may matter beyond pooled magnitude, but inspect realized removal counts and per-token distribution shifts.
- If **G differs from L**, it may reflect fixed versus per-token adaptive quota, **not** a different within-token raw-magnitude ranking.
- If all four are similar to random, these strength proxies add little at the tested layers, or the layer is insensitive; check effect sizes and all-zero reference.
- If results differ across GPT-2 and SmolLM2, the models differ in pretraining, architecture, depth, nonlinearities, and tokenization: don't attribute the difference to one feature.

Zero ablation can create unusual internal states. KL measures fidelity of the *intervened output distribution*, not uniqueness or human interpretability of an explanation. This is a study of activation-event **selection for zeroing**, not an independent neuron simulator or a proof that a learned circuit has been understood.

## References

- Leo Gao, *Neural networks are not already clean* (2026): https://www.lesswrong.com/posts/Dkh7pXKjoNWkkxQDz/neural-networks-are-not-already-clean
- Nick Cammarata, *Neural Networks Are Already Clean* (2026): https://already-clean.goodfire.com/
- GPT-2 pretrained checkpoint: https://huggingface.co/openai-community/gpt2
- SmolLM2-135M pretrained **base** checkpoint: https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- Official SmolLM2 config (`30` layers, `1536` MLP intermediate): https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config.json
- WikiText-2 raw: https://huggingface.co/datasets/Salesforce/wikitext
- Transformers GPT-2 implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
- Transformers Llama/SmolLM2 implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
- PyTorch module forward pre-hooks: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
