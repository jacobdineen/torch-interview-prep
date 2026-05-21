"""
Problem 04d: outer_product of 1D vectors

(Split from parent problem 04: Problem 04: Broadcasting and Arithmetic)
"""

import torch


def outer_product(u, v):
    return u.outer(v)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
