"""
Problem 04c: pairwise_squared_distances computes ||a-b||^2 over D

(Split from parent problem 04: Problem 04: Broadcasting and Arithmetic)
"""

import torch


def pairwise_squared_distances(A, B):
    a2 = (A * A).sum(dim=1, keepdim=True)  # (M, 1)
    b2 = (B * B).sum(dim=1, keepdim=True).T  # (1, N)
    return (a2 + b2 - 2 * A @ B.T).clamp(min=0)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
