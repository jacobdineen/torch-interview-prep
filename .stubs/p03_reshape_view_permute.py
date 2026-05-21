"""
Problem 03: Reshape, View, Permute

Implement:
  - flatten_batch(x)      : collapse all dims except dim 0  (B, C, H, W) -> (B, C*H*W)
  - swap_last_two(x)      : transpose the last two dims
  - to_channels_last(x)   : (B, C, H, W) -> (B, H, W, C)
  - require_contiguous(x) : return a contiguous tensor with the same data
"""

import torch


def flatten_batch(x):
    raise NotImplementedError


def swap_last_two(x):
    raise NotImplementedError


def to_channels_last(x):
    raise NotImplementedError


def require_contiguous(x):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
