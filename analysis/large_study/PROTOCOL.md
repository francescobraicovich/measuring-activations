# Expanded article-sampled study

This is a new exploratory evaluation after the earlier test results were inspected. It uses the same frozen GPT-2 and SmolLM2 base checkpoints and train-only calibration, with 512 deterministic nonoverlapping 128-token windows per model sampled round-robin across the same 61 eligible WikiText-2 test articles. Tokenizer-specific text spans can differ. Uncertainty uses 1,000 article-cluster bootstrap resamples, retaining within-article windows.

Evaluate GPT-2 layers 0, 2, 6, 8, 11 and SmolLM2 layers 0, 5, 16, 21, 29. At every layer, use nominal removal fractions 10%, 20%, 30%, 40%, 50%, 60%, 70%, 75%, 80%, 85%, 90%, 95%, 99%, 99.5%, 99.9%, and 100%, with identity as an additional reference. Apply all four original rankings and all four rankings of distance from the signed per-channel train mean. Zero the original activation in every intervention.

Use one shared random comparator per fraction, a nested prefix of a uniformly random per-token channel permutation. It exactly matches L and W event counts, but calibrated N/G can remove different counts; show actual removal fractions and do not call those comparisons strictly count-matched. Randomness uses a fixed seed and the intervals reflect article resampling, not variation over random seeds.

Calibrate from the train split, validate both models on validation text, and only then run the held-out sample. Record model, tokenizer, dataset and code provenance, saved masks' aggregate counts, removed signal, next-token KL, NLL change, and top-1 flip rates. Interventions cover all 128 positions and outcomes score positions 16–126. Report findings as exploratory and distinguish within-model layers from cross-model comparisons.
