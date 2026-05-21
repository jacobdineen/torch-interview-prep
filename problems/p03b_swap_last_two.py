"""
Problem 03b: swap_last_two transposes the last two dims

(Split from parent problem 03: Problem 03: Reshape, View, Permute)
"""

import torch


def swap_last_two(x):
    return x.transpose(-2, -1)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
