"""
Problem 27a: train: drop rate is roughly p

(Split from parent problem 27: Problem 23: Inverted Dropout From Scratch)
"""

import torch
import torch.nn as nn


class MyDropout(nn.Module):
    def __init__(self, p=0.5):
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
