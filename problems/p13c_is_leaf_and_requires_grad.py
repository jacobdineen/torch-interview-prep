"""
Problem 13c: is_leaf_and_requires_grad reports the right flags

(Split from parent problem 13: Problem 13: Controlling Gradient Flow)
"""

import torch
import torch.nn as nn


def is_leaf_and_requires_grad(t):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
