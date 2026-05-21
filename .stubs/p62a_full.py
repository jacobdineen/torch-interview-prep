"""
Problem 62a: full-pass output shape (B, T, D)

(Split from parent problem 62: Problem 46: Multi-Head Attention With KV Cache)
"""
import math
import torch
import torch.nn as nn

class CachedMHA(nn.Module):

    def __init__(self, d_model, num_heads):
        raise NotImplementedError

    def forward(self, x_new, cache=None):
        raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
