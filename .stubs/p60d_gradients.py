"""
Problem 60d: gradients are finite for every parameter

(Split from parent problem 60: Problem 53: SwiGLU FFN)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):

    def __init__(self, d_model, d_ff):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
