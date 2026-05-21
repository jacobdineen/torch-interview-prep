"""
Problem 10: Top-k, Sort, Argmax

  - top_k_per_row(x, k)             : (N, V) scores -> (values: (N, k), indices: (N, k))
                                       sorted in descending order
  - sort_then_take_indices(x, k)    : like top_k but returns only indices, ascending order
                                       (i.e., the k smallest)
  - argmax_2d(x)                    : (H, W) -> (row, col) of the global max as a 1D LongTensor of length 2
  - kth_largest(x, k)               : 1D tensor -> the k-th largest element (1-indexed)
"""

import torch

def top_k_per_row(x, k):
    raise NotImplementedError

def sort_then_take_indices(x, k):
    raise NotImplementedError

def argmax_2d(x):
    raise NotImplementedError

def kth_largest(x, k):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
