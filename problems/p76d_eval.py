"""
Problem 76d: eval mode bypasses checkpointing and still produces correct output

(Split from parent problem 76: Problem 76: Activation (Gradient) Checkpointing)
"""

import torch
import torch.nn as nn


class CheckpointedSequential(nn.Module):
    def __init__(self, blocks):
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
