"""
Problem 74a: gumbel_noise produces finite values

(Split from parent problem 74: Problem 55: Gumbel-Softmax (with Straight-Through Estimator))
"""
import torch

def gumbel_noise(shape, eps=1e-20, generator=None):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
