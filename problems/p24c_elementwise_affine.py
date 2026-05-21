"""
Problem 24c: elementwise_affine=False also matches

(Split from parent problem 24: Problem 22: LayerNorm From Scratch)
"""

import torch
import torch.nn as nn


class MyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
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
