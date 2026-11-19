# What is a weak activation?

An activation can be small for one neuron and unusually large for another. If we want to know which activations a language model can lose with little change to its predictions, what should we measure them against?

I compare four definitions of “weak” by removing the activations each one selects in GPT-2 and SmolLM2, then measuring how their next-token predictions change.

[Read the interactive article](https://francescobraicovich.github.io/measuring-activations/) · [Methods and reproduction](analysis/large_study/PROTOCOL.md)
