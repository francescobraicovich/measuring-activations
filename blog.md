October 2026

# What is a weak activation?

Comparing four definitions in GPT-2 and SmolLM2.

The interpretability community has seen considerable discussion about weak activations following Nick Cammarata’s [“Neural Networks Are Already Clean”](https://already-clean.goodfire.com/). Nick argues that modern training pipelines have improved neuron monosemanticity alongside model performance, and that interpreting neurons’ activations has become tractable even at large scale. In his essay, he also claims that vision models tolerate substantial removal of weak activations. It is unclear whether this also holds for LLMs, and most importantly: what is a weak activation?

Whether an activation is strong or weak makes sense only compared to a reference distribution: we might pick the neuron’s history, other neurons in the same layer, or its contribution to the MLP output. These comparisons can select quite different values for removal. With the following experiments, I try to navigate these differences.

## The Discussion

As mentioned before, [“Neural Networks Are Already Clean”](https://already-clean.goodfire.com/) argues that removing weak activations in vision models mostly affects fine distinctions. Throughout the essay, Nick frequently uses the notion of activation strength, sometimes looking at a neuron’s response relative to its own strongest observed activation. To my knowledge, it is not clear what definition he uses in the vision ablations, as he only describes removing the “weakest 75%”.

In this [LessWrong discussion](https://www.lesswrong.com/posts/x4kvabkF2Jk8GLZp8/linkpost-neural-networks-are-already-clean), Neel Nanda says he expects weak activations to matter for language-model behavior, including memorized facts, and doubts that Nick’s results will hold for LLMs.

## Selection rules

My experiments focus on the MLP layers of transformers. I compare four ways of defining what makes an activation weak, using different reference distributions and selecting approximately the lowest fraction $p$ of activations.

### N · Neuron percentile
 
For each neuron, remove activations below that neuron’s $p$-quantile of magnitudes on the training text.

### L · Layer percentile at token

For each token, remove the $\lfloor pD\rfloor$ neurons with the smallest activation magnitudes in that layer, where $D$ is the number of neurons.

### G · Layer percentile, globally pooled across tokens

Find a single threshold at the $p$-quantile of activation magnitudes pooled across all neurons and tokens in the layer’s training text. Remove activations below that threshold.

### W · Layer percentile, weight adjusted

For each token, remove the $\lfloor pD\rfloor$ neurons with the smallest activation magnitude multiplied by the norm of their output weight vector.

### Selections for the same token

Change an activation or its output-vector norm in this synthetic eight-neuron layer. The table shows which activations each rule selects; calibration histories stay fixed.

**\[Figure 1: interactive selection rules\]**

*Figure 1. Synthetic example. Each neuron has 12 calibration magnitudes, evenly spaced from 0.1 to 1.9 times its reference scale. N and G use linearly interpolated quantiles and strict $<$ thresholds; L and W use exact quotas with ties broken by neuron index. [Calculation.](analysis/large_study/strength_definitions.js)*

## Results

Choose a model, layer, outcome, ranking, and horizontal and vertical scales in the interactive plot.

**\[Figure 2: interactive ablation outcome plot\]**

Figure 2. Points use mean observed removal, which can differ from the nominal fraction. A logarithmic horizontal scale omits points at zero removal. Error bars are 95% article-cluster bootstrap intervals for the selected outcome. Hover for nominal fractions; use the legend to hide or show methods. Mean-centering changes the ranking score; selected activations are set to zero. Random is one shared comparator per nominal fraction. Its per-token count matches L and W exactly. Calibrated N and G thresholds can remove a different observed fraction, so their counts need not match random exactly.

## Removing weak activations

I tested the rules on frozen GPT-2 Small and SmolLM2-135M, using 512 WikiText-2 windows from 61 articles for each model. Each experiment zeroes activations in one of five sampled layers. The plot above shows the resulting KL and the fraction of positions where the most likely next token changes. The random baseline removes the same number of neurons as L and W at each token.

In SmolLM2, we can remove most activations with relatively little change to the model’s predictions. At the nominal 75% setting, G removes between 74% and 75% of activations across the five sampled layers. The most likely next token changes at 4.0–5.5% of scored positions, and KL ranges from 0.0044 to 0.0095. G has the lowest measured KL and flip rate of the four rules in every sampled SmolLM2 layer.

In layer 16, G removes 74.5% of activations, giving KL **0.00443** and a **4.1%** flip rate. Removing 75% at random gives KL 0.0496 and a 13% flip rate. G and random do not remove exactly the same number of activations; L matches random’s count and still gives a much lower KL of 0.0048.

Removing three quarters is not the end of the curve. In the same SmolLM2 layer, G gives KL 0.00062 at 50% nominal removal, 0.00443 at 75%, and 0.0143 at 90%. The corresponding flip rates are 1.53%, 4.1%, and 7.3%. The observed removal rates are 49.5%, 74.5%, and 89.7%. More removal changes the predictions more, but the increase is gradual in this layer.

**\[Figure 3: ablation results across layers\]**

*Figure 3. L, W, and random removal select exactly 75% of neuron activations at each token. Lines connect the five measured layers per model. Both panels use the same logarithmic KL scale. Whiskers show 95% article-cluster bootstrap intervals from 1,000 resamples; some are narrower than the symbols. Random removal uses one fixed seed; the intervals exclude variation across seeds.*

The result is different in GPT-2. At the nominal 75% setting, N has the lowest measured KL and flip rate of the four rules in all five sampled layers. In layer 6, N removes 75.1% of activations, with KL 0.0461 and a 14.3% flip rate. L removes exactly 75%, with KL 0.0796 and a 17.9% flip rate; its KL is slightly higher than random’s 0.076. For this paired comparison, L minus random is +0.0036 ± 0.0017 nats, with approximate 95% article-bootstrap uncertainty.

GPT-2’s removal curves also show that the cost need not increase steadily as more activations are zeroed. In layer 6, L reaches KL 0.13 at 90% nominal removal, then falls to 0.0851 at 99%. Removing additional activations changes the combined perturbation, and can partly cancel an earlier change. I would not interpret every increase in the removal fraction as an independent measurement of the activations added to the mask.

## Which definition should we use?

I would start with G when looking for activations that can be removed. It performs best on both measured outcomes in every sampled SmolLM2 layer, and it has a simple interpretation: it removes activations that are small compared with the layer’s pooled training distribution. It lets the number removed vary across tokens instead of imposing the same quota on each one.

I would also compare it with N. A small activation may be ordinary for one neuron and unusual for another. N uses each neuron’s history, and in GPT-2 that gives lower KL and flip rates than the other three rules in every sampled layer. The better reference changes between these two models.

The difference can be large. In SmolLM2’s final layer, G removes 74.3% of activations and gives KL 0.007. N removes 73.9% and gives KL 0.254. N can remove an activation that is large compared with other neurons because it is small for that particular neuron. The two rules also distribute removals differently across tokens, so their similar aggregate removal rates do not mean that they select the same activations.

## Weighting by the MLP output

The MLP output projection is a matrix: each neuron has a vector of weights, one for each dimension of the MLP output. Multiplying that vector by the neuron’s activation gives its contribution to the output. W uses the L2 norm of this whole contribution, $\lvert a_j\rvert\,\lVert\mathbf{w}_j\rVert_2$, to give each neuron one score.

A small activation can still have a large effect on the MLP output if its output weight vector is large. W accounts for that by ranking the length of each neuron’s contribution. I expected that to be a useful way to decide which activations to remove, but in GPT-2’s first layer it performs considerably worse than L.

At 75% removal, W gives KL **0.814** and a 44.5% flip rate. L gives KL 0.139 and a 19.1% flip rate. W removes less of the summed individual contribution lengths (47.9% versus 49.5%), yet the combined effect on predictions is much larger.

One reason is cancellation. Suppose three neurons contribute 1, −1, and 0.9 in the same output direction. If we remove the two smallest contributions by length, W removes 0.9 and one of the others, leaving a removed sum of either 1.9 or −0.1 depending on the tie break. Removing 1 and −1 instead gives a sum of zero. Looking at each contribution’s length alone misses the way their directions interact.

**\[Figure 4: projection geometry plot\]**

*Figure 4. GPT-2 layer 0, 75% removal, on an earlier sample of 128 windows from four articles. Geometry is computed from intact activations at the output projection and averaged over positions 16–126, then windows. Left: the sum of individual contribution lengths. Right: the length of the removed vector after summing contributions. The axes use different scales and start at zero. Whiskers resample only four articles. W centered ranks distance from the training mean, then zeroes the original activation.*

In this smaller sample, W reduces the sum of individual contribution lengths from 481 to 465, but increases the length of the combined removed vector from 12.1 to 16.6. The difference appears at the MLP output itself, before considering the effect of the remaining layers. The diagnostic does not separate the downstream effects of the perturbation’s length and direction.

Changing the reference can help here too. If W ranks distance from each neuron’s signed training mean instead of distance from zero, its first-layer GPT-2 KL falls to 0.133, close to L’s 0.139. This only changes how activations are selected; the selected values are still replaced with zero.

## What this says about weak activations

These experiments make me more sympathetic to Nick’s view of weak activations. In SmolLM2, removing roughly three quarters of them with a pooled magnitude threshold changes the most likely next token only about one time in twenty. The same fraction removed at random is much more disruptive. A large amount of activity can be removed while leaving the model’s predictions relatively close to the original.

For deciding what to remove, I would start with pooled magnitude and check a neuron-specific threshold alongside it. G works best in the sampled SmolLM2 layers; N works best in the sampled GPT-2 layers. Weighting by output size captures something the scalar magnitude misses, but it does not account for cancellation or the layers that follow.

This is evidence that many weak activations can be removed at low average cost in these models. It does not establish that the removed responses are semantically coherent, or that newer models are cleaner because of their training. The experiment changes one layer at a time, uses one corpus and two small language models, and could miss rare cases that matter despite their small effect on the average. I would next repeat the random control with more seeds, compare zero and mean replacement under the same masks, and test on text chosen to include difficult factual cases.

## Methods and data

**Models and hook.** Frozen base checkpoints `openai-community/gpt2` (~124M parameters; layers 0, 2, 6, 8, 11) and `HuggingFaceTB/SmolLM2-135M` (layers 0, 5, 16, 21, 29). Hooks modify the input to GPT-2’s `mlp.c_proj`, after GELU, and SmolLM2’s `mlp.down_proj`, after the full $\operatorname{SiLU}(\mathrm{gate})\odot\mathrm{up}$ product. Projection biases remain in place. Model and dataset revisions, calibration hashes, and source hashes are recorded with the results.

**Data and selection.** WikiText-2 raw, 512 nonoverlapping 128-token windows per model, selected in deterministic article rounds from the same 61 eligible test articles. Article multiplicities match across models; tokenizers produce different spans. Means and N/G thresholds use 128 training windows. Each intervention touches one layer over positions 0–127; predictions at positions 16–126 are scored against the following tokens, giving 56,832 scored positions per model per condition.

**Masks and controls.** Nominal fractions: 10, 20, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 99, 99.5, 99.9, and 100%, plus identity. N/G use NumPy’s default linear quantile interpolation and strict $<$; ties and distribution shift can change realized counts. L/W use stable sorting and $\lfloor pD\rfloor$. At 0% and 100% the mask is explicitly empty or full. Random uses nested prefixes of one seeded neuron permutation per token, matching L/W counts exactly. N/G counts can differ. The centered variants change selection only; selected original values are zeroed.

**Estimates and uncertainty.** Points are window-weighted means. Bootstrap uncertainty uses 1,000 resamples of articles, keeping each sampled article’s windows together; paired comparisons use within-window differences before resampling. Numerical ± values give the mean and half the width of the 95% percentile interval, rounded for readability. They are approximate because the interval may not be symmetric about the mean. Plot whiskers show the original percentile bounds. Uncertainty is pointwise, unadjusted for the many inspected conditions, and conditional on one random-control seed.

**Scope.** Earlier results from this test corpus were inspected during study design. The geometry diagnostic uses an earlier four-article subset and is not an independent replication of the larger study. The evidence covers two small language models, one corpus, and interventions at one layer at a time. Average KL can conceal rare-case losses, and zero replacement can create internal states uncommon in the intact model.

### Output weighting and cancellation

Write the activation of neuron $j$ as $a_j$ and its output vector as $\mathbf{w}_j$. For a set $S$ of removed neurons, the MLP output perturbation satisfies

$$
\delta\mathbf{h}=-\sum_{j\in S}a_j\mathbf{w}_j,\qquad \lVert\delta\mathbf{h}\rVert_2\leq\sum_{j\in S}\lvert a_j\rvert\lVert\mathbf{w}_j\rVert_2.
$$

At a fixed removal count, W minimizes the right-hand side. The bound omits cancellation between output vectors, so it need not minimize the actual perturbation. W is invariant to multiplying a projection-input coordinate by a positive constant and dividing its output vector by the same constant; L need not be.

The saved-data build needs no model inference. It regenerates the figures and checks numerical statements against the tables. Sources linked above were consulted on 30 October 2026.

    python analysis/large_study/visualize.py
    python analysis/large_study/write_report.py
    python -m pytest -q

Use the scientific Python environment in [requirements.txt](requirements.txt); Node.js is also needed to typeset the mathematics and run the JavaScript/Python mask-parity test. Full model-evaluation commands are in the [README](README.md). The original [planning document](plan.md) is historical; the [expanded-study protocol](analysis/large_study/PROTOCOL.md) describes the reported sweep.

[Sample manifest](results/large_study/sample_manifest.json)[Per-window measurements](results/large_study/data/metrics.parquet)[Summary CSV](results/large_study/data/summary.csv)[Paired differences](results/large_study/data/paired_comparisons.csv)[Projection geometry](results/followup/localization/geometry.parquet)[Diagnostic provenance](results/followup/localization/manifest.json)[Standalone explorer](results/large_study/interactive.html)

## More results

### How the selected outcome varies across layers

Uses the selected model, outcome, ranking and nominal fraction. Hover for observed removal; calibrated methods can have different realized budgets. The lines connect only the five sampled layers. A logarithmic vertical scale can reveal small differences, but does not display zero or negative values.

**\[Interactive cross-layer comparison plot\]**

**Methods:** N selects activations below a per-neuron training threshold; G uses one global training threshold. L selects the smallest magnitudes within each token; W selects the smallest magnitude multiplied by the outgoing weight norm. Random selects neurons uniformly. Identity removes none; all zero removes all.

## Cross-layer comparison

Each cell shows mean KL ± approximate 95% article-bootstrap uncertainty and observed removal for the selected nominal fraction. Identity and all-zero controls appear only at their measured fractions.

## Paired comparisons

For the selected model, layer and nominal fraction, each difference is method minus reference, computed within the same windows. Bootstrap resampling preserves article clusters and pairing. Observed removal differences show whether the actual removal rates match.

## Measurements by fraction and method

Numerical ± values are the mean and half the width of the 95% article-bootstrap percentile interval; they approximate the potentially asymmetric bounds shown by the plot whiskers. Random repetitions, if present, are averaged within each window before summarization. Values reflect the available study output.
