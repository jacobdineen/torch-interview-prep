# Build Your Own Autograd Engine (micrograd)

Build a scalar reverse-mode automatic differentiation engine — the core idea
behind PyTorch — and then train a neural net with it. Every number is a `Value`
node that remembers how it was computed; each operation wires up a local
`_backward` closure, and one topological pass propagates gradients from a scalar
loss back to every leaf. **18 steps, 5 parts.**

| Part | Focus |
|------|-------|
| 1 | Scalar ops: add, mul, pow, sub (each with its local gradient) |
| 2 | Activations: ReLU, tanh, exp |
| 3 | Reverse-mode autodiff: topological sort + backprop |
| 4 | Neurons, layers, an MLP — all out of Value nodes |
| 5 | Parameters, MSE loss, zero_grad, an SGD step, the training loop |

The `Value` class is provided; you implement every operation and the backward
pass. Tests check each gradient against a numerical finite-difference oracle.

```bash
uv run python projects.py micrograd-autograd-from-scratch            # parts + steps
uv run python projects.py micrograd-autograd-from-scratch --scaffold # grad check + train an MLP
```
