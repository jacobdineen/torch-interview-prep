"""
Problem 61a: output shape matches input

(Split from parent problem 61: Problem 61: GPT-Style Decoder Block)
"""
import torch
import torch.nn as nn

class GPTDecoderBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
