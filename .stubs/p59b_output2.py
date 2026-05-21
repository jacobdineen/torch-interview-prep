"""
Problem 59b: output is non-trivially different from input

(Split from parent problem 59: Problem 44: Transformer Encoder Block (post-LN))
"""
import torch
import torch.nn as nn

class TransformerEncoderBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        raise NotImplementedError

    def forward(self, x, attn_mask=None):
        raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
