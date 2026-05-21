"""
Problem 05c: normalize_along_dim subtracts mean and divides by std along dim

(Split from parent problem 05: Problem 05: Reductions Along Dimensions)
"""
import torch

def normalize_along_dim(x, dim):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
