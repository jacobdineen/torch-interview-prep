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
    return torch.flatten(x, start_dim=1)


def swap_last_two(x):
    return x.transpose(-2, -1)


def to_channels_last(x):
    print(x.shape)
    print(x.transpose(1, -1).shape)
    # 2,3,4,5 -> 2,4,5,3

    return x.permute(0, 2, 3, 1)


def require_contiguous(x):
    return x.contiguous()


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
