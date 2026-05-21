"""
Problem 10c: argmax_2d returns the (row, col) of the global max

(Split from parent problem 10: Problem 10: Top-k, Sort, Argmax)
"""
import torch

def argmax_2d(x):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
