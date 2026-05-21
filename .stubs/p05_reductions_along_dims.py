"""
Problem 05: Reductions Along Dimensions

Practice sum/mean/max/argmax with dim and keepdim.

  - per_channel_mean(x): (N, C, H, W) -> (C,) mean across batch & spatial dims
  - per_batch_max(x)   : (B, *)       -> (B,) max over all non-batch dims
  - normalize_along_dim(x, dim): subtract mean and divide by population std along dim
  - rowwise_argmax(x)  : (N, K)       -> (N,) indices of max along last dim
"""

import torch


def per_channel_mean(x):
    raise NotImplementedError


def per_batch_max(x):
    raise NotImplementedError


def normalize_along_dim(x, dim):
    raise NotImplementedError


def rowwise_argmax(x):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
