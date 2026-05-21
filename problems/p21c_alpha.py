"""
Problem 21c: alpha=[0.25]*4 scales loss by 0.25 vs alpha=None

(Split from parent problem 21: Problem 20: Focal Loss (multi-class))
"""

import torch
import torch.nn.functional as F


def focal_loss(logits, targets, gamma=2.0, alpha=None, reduction="mean"):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
