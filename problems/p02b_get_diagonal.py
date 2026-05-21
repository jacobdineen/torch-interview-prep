"""
Problem 02b: get_diagonal returns the matrix diagonal

(Split from parent problem 02: Problem 02: Indexing and Slicing)
"""

import torch


def get_diagonal(x):
    return torch.diag(x)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
