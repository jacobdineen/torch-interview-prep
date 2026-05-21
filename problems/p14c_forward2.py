"""
Problem 14c: forward on (2, 6, 8) batched input matches nn.Linear

(Split from parent problem 14: Problem 14: Linear Layer From Scratch)
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
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
