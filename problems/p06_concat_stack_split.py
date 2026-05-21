"""
Problem 06: Concatenation, Stacking, Splitting

Understand the difference between cat (joins along existing dim) and stack (new dim).

  - cat_features(a, b)        : (N, D1) and (N, D2)   -> (N, D1+D2)   along feature dim
  - stack_batch(items)        : list of K tensors (D,) -> (K, D)
  - split_evenly(x, k)        : split tensor of size N along dim 0 into k equal chunks (tuple)
  - interleave(a, b)          : (N, D), (N, D)         -> (2N, D) rows alternating a[0], b[0], a[1], b[1], ...
"""

import torch


def cat_features(a, b):
    return torch.concatenate((a, b), dim=-1)


def stack_batch(items):
    return torch.stack(items, dim=-1)


def split_evenly(x, k):
    return torch.split(x, k)


def interleave(a, b):
    return torch.repeat_interleave(a, b)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
