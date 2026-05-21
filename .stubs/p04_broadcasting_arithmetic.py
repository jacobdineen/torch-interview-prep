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
    raise NotImplementedError


def normalize_per_sample(X, eps=1e-6):
    raise NotImplementedError


def pairwise_squared_distances(A, B):
    raise NotImplementedError


def outer_product(u, v):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
