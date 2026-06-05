"""Build an MLP in JAX from Scratch — end-to-end demo.

    python projects.py mlp-in-jax-from-scratch --scaffold

Generates a synthetic, linearly-separable classification dataset, builds an MLP
from your step functions, trains it with jax.grad + functional SGD, and reports
accuracy. Pure CPU JAX, well under a minute.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
from solution import *  # noqa: E402,F401,F403


def main():
    in_dim, hidden, n_classes, n = 8, [32, 16], 4, 512
    key = make_prng_key(0)
    kx, ky, kp = split_prng_key(key, 3)

    X = sample_input_features(kx, n, in_dim)
    labels = assign_class_labels(X, n_classes)
    Y = one_hot_encode_labels(labels, n_classes)
    params = init_mlp_params(kp, [in_dim] + hidden + [n_classes])

    print(f"mlp {[in_dim] + hidden + [n_classes]}  n={n} classes={n_classes}")
    params, history = train_mlp(params, X, Y, lr=0.3, n_steps=300)
    acc = float(classification_accuracy(mlp_forward(params, X), labels))
    print(f"loss: {history[0]:.3f} -> {history[-1]:.3f}   train accuracy: {acc:.3f}")
    print("ok" if history[-1] < history[0] and acc > 0.8 else "warning: did not converge")


if __name__ == "__main__":
    main()
