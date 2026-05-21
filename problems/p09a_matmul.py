"""
Problem 09a: matmul: (M,K) @ (K,N) -> (M,N)

(Split from parent problem 09: Problem 09: einsum Practice)
"""

import torch


def matmul(A, B):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
