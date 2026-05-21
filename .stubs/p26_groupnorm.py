"""
Problem 52: GroupNorm

Split channels into `num_groups` groups, normalize within each group across all
spatial positions, then apply per-channel affine.

  - MyGroupNorm(num_groups, num_channels, eps=1e-5, affine=True)
        * For input (N, C, H, W) (or any (N, C, *) shape), reshape channels into
          (N, num_groups, C/num_groups, *), compute mean and population variance
          along all axes except (N, num_groups), then normalize.
        * If affine: learnable weight (init 1) and bias (init 0) per channel.

Match torch.nn.GroupNorm.
"""

import torch
import torch.nn as nn


class MyGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
