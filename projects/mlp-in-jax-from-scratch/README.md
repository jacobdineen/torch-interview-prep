# Build an MLP in JAX from Scratch

Implement a multi-layer perceptron end to end in **JAX**, from functional PRNG-key
handling and parameter initialization through forward passes, loss, autodiff, and
SGD training. By the end you have a working classifier trained with `jax.grad` and
pure functional updates — 21 steps across 7 parts.

| Part | Focus |
|------|-------|
| 1 | Functional PRNG: keys, splits, sampling |
| 2 | Synthetic features, labels, one-hot |
| 3 | Parameter initialization (`[(W,b), ...]` pytree) |
| 4 | Linear / ReLU / softmax / MLP forward |
| 5 | Log-softmax, cross-entropy, accuracy |
| 6 | `jax.grad` autodiff + functional SGD |
| 7 | Training step, loop, and prediction |

```bash
uv run python projects.py mlp-in-jax-from-scratch            # parts + steps
uv run python projects.py mlp-in-jax-from-scratch --next     # next unsolved step
uv run python projects.py mlp-in-jax-from-scratch --scaffold # end-to-end demo (once solved)
```
