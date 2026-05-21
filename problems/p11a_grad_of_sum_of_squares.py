"""
Problem 11a: grad_of_sum_of_squares matches 2*x

(Split from parent problem 11: Problem 11: Autograd Basics)
"""

import torch


def grad_of_sum_of_squares(x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
