"""
Problem 03a: flatten_batch collapses all dims except batch

(Split from parent problem 03: Problem 03: Reshape, View, Permute)
"""
import torch

def flatten_batch(x):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
