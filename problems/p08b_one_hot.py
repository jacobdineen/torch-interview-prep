"""
Problem 08b: one_hot makes a (N, C) float matrix

(Split from parent problem 08: Problem 08: Gather and Scatter)
"""

import torch


def one_hot(labels, num_classes):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
