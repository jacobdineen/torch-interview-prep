"""
Problem 02c: every_other_col returns columns 0, 2, 4, ...

(Split from parent problem 02: Problem 02: Indexing and Slicing)
"""

import torch


def every_other_col(x):
    return x[:, ::2]


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
