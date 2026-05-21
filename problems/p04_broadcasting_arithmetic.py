"""
Problem 04: Broadcasting and Arithmetic

Use broadcasting (no Python loops, no .repeat / .expand_as in the body unless natural):

  - center_rows(X)                       : subtract per-row mean  (X: (N, D))
  - normalize_per_sample(X, eps=1e-6)    : per-row mean 0, std 1 (population std)
  - pairwise_squared_distances(A, B)     : (M, D), (N, D) -> (M, N) of ||a-b||^2
  - outer_product(u, v)                  : 1D u and v -> 2D outer product
"""

import torch


def center_rows(X):
    return X - X.mean(dim=1, keepdim=True)


def normalize_per_sample(X, eps=1e-6):
    mu = X.mean(dim=1, keepdim=True)
    std = X.std(unbiased=False, dim=1, keepdim=True)
    return (X - mu) / (std - eps)


def pairwise_squared_distances(A, B):
    a2 = (A * A).sum(dim=1, keepdim=True)  # (M, 1)
    b2 = (B * B).sum(dim=1, keepdim=True).T  # (1, N)
    return (a2 + b2 - 2 * A @ B.T).clamp(min=0)  # (M, N)


def outer_product(u, v):
    return u.outer(v)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
