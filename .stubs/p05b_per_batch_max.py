"""
Problem 05b: per_batch_max takes the max over all non-batch dims

(Split from parent problem 05: Problem 05: Reductions Along Dimensions)
"""
import torch

def per_batch_max(x):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
