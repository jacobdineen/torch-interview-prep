"""
Problem 23b: running stats updated like nn.BatchNorm1d

(Split from parent problem 23: Problem 21: BatchNorm1d From Scratch)
"""

import torch
import torch.nn as nn


class MyBatchNorm1d(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5, affine=True):
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
