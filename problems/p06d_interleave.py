"""
Problem 06d: interleave alternates rows from a and b

(Split from parent problem 06: Problem 06: Concatenation, Stacking, Splitting)
"""

import torch


def interleave(a, b):
    return torch.repeat_interleave(a, b)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
