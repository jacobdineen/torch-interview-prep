"""
Problem 04b: normalize_per_sample yields per-row mean 0 std 1

(Split from parent problem 04: Problem 04: Broadcasting and Arithmetic)
"""
import torch

def normalize_per_sample(X, eps=1e-06):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
