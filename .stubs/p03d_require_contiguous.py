"""
Problem 03d: require_contiguous returns a contiguous tensor with same data

(Split from parent problem 03: Problem 03: Reshape, View, Permute)
"""
import torch

def require_contiguous(x):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
