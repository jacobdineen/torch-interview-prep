"""
Problem 14: Linear Layer From Scratch

Implement an nn.Module equivalent to nn.Linear without using F.linear or nn.Linear.

  - MyLinear(in_features, out_features, bias=True)
        * Weight shape: (out_features, in_features), initialized with Kaiming uniform
          (same default as nn.Linear).
        * Bias shape: (out_features,), initialized uniformly in [-1/sqrt(in_features), +1/sqrt(in_features)]
          (same default as nn.Linear).
        * forward(x): supports inputs of shape (*, in_features) and produces (*, out_features).

You may use torch.nn.init helpers and a plain matmul.
"""

import math
import torch
import torch.nn as nn

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
