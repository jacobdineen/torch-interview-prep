"""
Problem 10d: kth_largest picks the k-th largest (1-indexed) in a flat tensor

(Split from parent problem 10: Problem 10: Top-k, Sort, Argmax)
"""
import torch

def kth_largest(x, k):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
