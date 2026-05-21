"""
Problem 71d: matches the symmetric cross-entropy reference

(Split from parent problem 71: Problem 50: InfoNCE / SimCLR-Style Contrastive Loss)
"""

import math
import torch
import torch.nn.functional as F


def info_nce_loss(a, b, tau=0.1):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
