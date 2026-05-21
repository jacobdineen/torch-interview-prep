"""
Problem 05d: rowwise_argmax returns the index of max in each row

(Split from parent problem 05: Problem 05: Reductions Along Dimensions)
"""

import torch


def rowwise_argmax(x):
    return x.argmax(dim=-1)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
