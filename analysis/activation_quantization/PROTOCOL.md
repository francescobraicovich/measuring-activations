# SD-band activation quantization

This exploratory follow-up tests the proposed increasing bin density against
uniform activation quantization. It does not use SmoothQuant, change weights,
or measure an optimized integer implementation.

## Quantizers

Use G's reference population: pool MLP output-projection input activations across
neurons and training tokens separately within each layer. Compute the population
standard deviation of the signed values. Keep zero as the origin, without mean
subtraction, and preserve signs. This shares G's pooling convention; SD bands are
a new construction rather than the original G percentile thresholds.

For magnitudes in [k sigma, (k+1) sigma], target a quantization interval density
proportional to 2^k. Thus the first three full bands receive n, 2n, 4n intervals.
Normalize this density over [0, C] to the positive level budget, place equally
spaced coordinates in its integral, and invert those coordinates to construct
the levels. The normalization can make n fractional; at finite precision the
integer counts per band approximate the ratios. With C=3 sigma, 4-bit
quantization has exactly 1, 2, 4 positive intervals in those bands.

For b in {4, 6, 8}, use 2^(b-1)-1 positive nonzero levels, their negatives, and
zero: 2^b-1 distinct values, as in a symmetric signed integer quantizer. Both
methods have identical level counts. Assign the nearest reconstruction level;
ties go toward zero. Values outside the range saturate at the endpoints.

Compare four methods:

- `uniform_max`: ordinary symmetric uniform levels over the training absolute
  maximum, one static range per layer.
- `uniform_mse`: uniform levels with clipping selected by calibration activation
  MSE.
- `g_geometric`: the increasing-density SD-band levels, also with clipping
  selected by calibration activation MSE.
- `uniform_matched`: uniform levels with exactly the clipping range selected for
  `g_geometric`, isolating the effect of bin placement.

For both MSE-selected methods, search 1, 2, 3, 4, 6, 8, 12, 16, 24, 32 sigma and
the full training maximum, capped at that maximum with duplicates removed. The
same grid and sample are used for both. Each layer and bit width is fitted
separately. No held-out outcome selects clipping or any other parameter.

## Calibration and evaluation

Use the existing frozen GPT-2 and SmolLM2-135M checkpoints, cached at the commits
recorded by the original study. Calibrate on the first 128 nonoverlapping
128-token WikiText-2 training windows, using the existing data loader. Calculate
moments and maxima from all activations, and retain a deterministic uniform
sample of up to 4,096 values per layer per calibration batch for clipping MSE.
Record model, tokenizer, dataset, code and calibration provenance. Calibration
sample selection uses a fixed seed and does not depend on method or bit width.

Validate identity-hook parity and all method/bit/scope conditions on eight
validation windows before held-out evaluation. Compare the middle layer alone
(GPT-2 layer 6, SmolLM2 layer 16) and every MLP output projection in the same
forward pass. Later layers receive the results of previous interventions.
All other activations and weights remain float32. These are simulated
quantization/dequantization interventions, not bit-packed execution.

Evaluate the first 64 windows of the previous article-balanced WikiText-2 test
sample, checking the existing manifest's text and token hashes. This includes
61 articles. Modify all 128 positions and score next-token positions 16--126,
using the existing KL, baseline NLL, NLL change, and top-token flip metrics.
Save per-window measurements and paired method-minus-reference differences.
Use 1,000 article-cluster bootstrap replicates with fixed seeds for intervals.
Perplexity is exp(mean NLL) over these scored positions and is not a full-context
WikiText benchmark score.

The reference comparisons are `g_geometric` minus each uniform method. Intervals
cover article variation, not model, calibration, or domain variation. The study
is exploratory and uses small models and one corpus. Any memory or speed claim
would require a separate encoding and kernel experiment.
