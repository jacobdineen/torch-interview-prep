"""
Problem 55b: MHA is permutation-equivariant under input permutation

(Split from parent problem 55: Problem 42: Multi-Head Attention (Standalone))
"""

import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, bias=True):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, mask=None):
        raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
