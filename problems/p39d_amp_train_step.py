"""
Problem 39d: amp_train_step reduces loss over 50 iterations

(Split from parent problem 39: Problem 62: Mixed Precision With autocast)
"""

import torch
import torch.nn as nn


def forward_autocast(model, x, dtype=torch.bfloat16, device_type="cpu"):
    raise NotImplementedError

def amp_train_step(model, optimizer, x, y, dtype=torch.bfloat16, device_type="cpu"):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
