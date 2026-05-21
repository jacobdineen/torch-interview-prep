"""
Problem 02: Indexing and Slicing

Implement (no Python loops):
  - get_row(x, i)             -> i-th row of a 2D tensor
  - get_diagonal(x)           -> diagonal of a square matrix
  - every_other_col(x)        -> columns 0, 2, 4, ...
  - select_rows(x, idx)       -> rows in idx (a 1D LongTensor)
  - top_left_block(x, k)      -> the top-left (k, k) block of a 2D tensor
"""

import torch


def get_row(x, i):
    raise NotImplementedError


def get_diagonal(x):
    raise NotImplementedError


def every_other_col(x):
    raise NotImplementedError


def select_rows(x, idx):
    raise NotImplementedError


def top_left_block(x, k):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
