"""
Problem 59a: output shape matches input

(Split from parent problem 59: Problem 44: Transformer Encoder Block (post-LN))
"""

import torch
import torch.nn as nn


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, attn_mask=None):
        raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
