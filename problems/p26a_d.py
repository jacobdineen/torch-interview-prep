"""
Problem 26a: 4D input (N,C,H,W) matches nn.GroupNorm

(Split from parent problem 26: Problem 52: GroupNorm)
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
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
