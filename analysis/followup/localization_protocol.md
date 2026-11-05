# Exploratory localization of the GPT-2 first-layer effect

Selected after inspecting the original results; these analyses are exploratory. Use the original 128 held-out windows, GPT-2 layer 0, the existing train-only calibration, and p = 0.75. Compare L, W, and W centered; centered selection still replaces the original activation with zero.

For each rule, apply its original mask only to positions 0–15 (prefix), and separately only to positions 16–127 (suffix). Evaluate KL, NLL change, and top-1 flips at prediction positions 16–126. Reuse the previously saved all-position result. Compare methods within each positional regime; prefix and suffix remove different numbers of events, and their effects need not add. Use 1,000 paired article-cluster bootstrap resamples, reporting the four-article limitation.

On unmodified layer-0 activations, for each method and token calculate the net removed projection vector `(a * mask) @ c_proj.weight` (projection bias cancels) and its Euclidean norm, alongside `sum_j |a_j| ||w_j||`. Summarize means over prefix 0–15 and scored positions 16–126 per window, then paired method differences by article. This is a local geometric diagnostic, not causal mediation; channel cancellation, direction, and downstream sensitivity are not identified by a norm alone. No channels, positions, or thresholds are selected after viewing these results.
