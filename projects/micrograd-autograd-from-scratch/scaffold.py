"""Build Your Own Autograd Engine (micrograd) — end-to-end demo.

    python projects.py micrograd-autograd-from-scratch --scaffold

Shows the engine doing two things PyTorch does: (1) a reverse-mode gradient that
matches a numerical finite-difference check, and (2) training a small MLP built
entirely from Value nodes to fit a tiny dataset. Pure Python, no deps.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solution import *  # noqa: F401,F403


def main():
    # (1) Gradient check: f = (a*b + tanh(c))**2, compare autodiff vs finite diff.
    a, b, c = Value(1.5), Value(-2.0), Value(0.5)
    f = v_pow(v_add(v_mul(a, b), v_tanh(c)), 2)
    backward(f)

    def fval(av, bv, cv):
        import math
        return (av * bv + math.tanh(cv)) ** 2

    eps = 1e-6
    num_a = (fval(1.5 + eps, -2.0, 0.5) - fval(1.5 - eps, -2.0, 0.5)) / (2 * eps)
    print("gradient check  df/da: autodiff=%.5f  numeric=%.5f  match=%s"
          % (a.grad, num_a, abs(a.grad - num_a) < 1e-4))

    # (2) Train an MLP (2 -> 4 -> 4 -> 1) to fit a tiny dataset.
    layers = [[init_neuron(2, seed=i) for i in range(4)],
              [init_neuron(4, seed=10 + i) for i in range(4)],
              [init_neuron(4, seed=20)]]
    X = [[2.0, 1.0], [-1.0, -2.0], [1.0, -1.0], [-2.0, 2.0]]
    Y = [1.0, -1.0, 1.0, -1.0]
    losses = train(layers, X, Y, lr=0.1, n_steps=200)
    preds = [mlp_forward(layers, [Value(xi) for xi in x])[0].data for x in X]
    print("MLP train  loss %.4f -> %.4f  (%d params)"
          % (losses[0], losses[-1], len(parameters(layers))))
    print("  targets:    ", [round(y, 2) for y in Y])
    print("  predictions:", [round(p, 2) for p in preds])
    ok = (abs(a.grad - num_a) < 1e-4
          and losses[-1] < 0.1 * losses[0]
          and all((p > 0) == (y > 0) for p, y in zip(preds, Y)))
    print("ok" if ok else "warning: weak fit")


if __name__ == "__main__":
    main()
