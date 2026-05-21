"""
Problem 25a: weight has shape (D,) initialized to ones

(Split from parent problem 25: Problem 51: RMSNorm)
"""
import torch
import torch.nn as nn

class RMSNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-06):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
