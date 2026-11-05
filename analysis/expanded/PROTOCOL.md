# Exploratory extension

This extension was chosen after the original held-out results were inspected; it is not an independent confirmation. Keep the original 128 train and 128 test windows, one-layer zero interventions, score positions 16–126, and four-article clustered uncertainty.

- Primary middle layers: GPT-2 6 and SmolLM2 16. Removal fractions: 0, 0.1 through 0.9 in 0.1 steps, 0.25, 0.75, 0.95, 0.99, and 1. Retain the original four rules and add four centered counterparts.
- Centered score: `abs(a[t,j] - mean_train(a[:,j]))`; the weighted version multiplies by the outgoing norm. N/G use centered train-only thresholds; L/W rank centered scores per token. The intervention still replaces the original activation with zero, not its mean. Record actual removed raw and centered signal separately.
- Every rule uses per-token count-matched random controls. At 0.75 in the middle layers, use three seeds; elsewhere use one. Endpoints are exact. Evaluate the same inputs with paired window comparisons and 1,000 article-bootstrap replicates. Do not interpret four clusters as precise population uncertainty.
- Existing early/final layers retain the 0.75 depth check. If time allows, inspect predetermined additional relative depths: GPT-2 0, 4, 8, 10 and SmolLM2 0, 10, 21, 26. Use all eight rules and random controls at 0.75 with one seed. Report as exploratory appendix evidence.
- Compute signed means, thresholds, and descriptive summaries from train only; complete validation on both models before new test sweeps. Preserve original artifacts and compare repeated raw conditions against the original run. Selection of further analyses after viewing results is exploratory and should be disclosed.
