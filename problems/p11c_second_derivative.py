"""
Problem 11c: second_derivative of sum(x^4) equals 12*x^2

(Split from parent problem 11: Problem 11: Autograd Basics)
"""

import torch


def second_derivative(x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
