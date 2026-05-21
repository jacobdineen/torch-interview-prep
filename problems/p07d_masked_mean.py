"""
Problem 07d: masked_mean averages over true positions of the mask

(Split from parent problem 07: Problem 07: Boolean Masking and torch.where)
"""

import torch


def masked_mean(x, mask):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
