"""
Problem 01b: make_range returns a half-open arithmetic range

(Split from parent problem 01: Problem 01: Tensor Basics)
"""

import os
import sys
import torch


def make_range(start, end, step=1):
    return torch.tensor([i for i in range(start, end, step)])


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
