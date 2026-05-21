"""
Problem 58b: w_q weight is (H*head_dim, D); w_k/w_v are (G*head_dim, D)

(Split from parent problem 58: Problem 58: Grouped-Query / Multi-Query Attention)
"""

import math
import torch
import torch.nn as nn


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_q_heads, num_kv_groups, bias=False):
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
