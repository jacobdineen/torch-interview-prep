"""
Problem 15b: leaky_relu matches F.leaky_relu(0.1)

(Split from parent problem 15: Problem 15: Activation Functions From Scratch)
"""

import math
import torch


def leaky_relu(x, negative_slope=0.01):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
