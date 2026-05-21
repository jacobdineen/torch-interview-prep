"""
Problem 07: Boolean Masking and torch.where

  - clip_negatives(x)             : replace negatives with 0 (use torch.where, no relu)
  - count_above(x, threshold)     : number of elements strictly greater than threshold
  - replace_nan(x, value)         : replace NaN entries with value
  - masked_mean(x, mask)          : mean of x[mask]; mask is a bool tensor with x's shape
"""

import torch

def clip_negatives(x):
    raise NotImplementedError

def count_above(x, threshold):
    raise NotImplementedError

def replace_nan(x, value):
    raise NotImplementedError

def masked_mean(x, mask):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
