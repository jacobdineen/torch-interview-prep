"""
Problem 12a: forward matches torch.nn.functional.softplus over a wide range

(Split from parent problem 12: Problem 12: Custom Autograd Function)
"""

import torch
from torch.autograd import Function


def stable_softplus(x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
