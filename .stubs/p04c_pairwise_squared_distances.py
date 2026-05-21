"""
Problem 04c: pairwise_squared_distances computes ||a-b||^2 over D

(Split from parent problem 04: Problem 04: Broadcasting and Arithmetic)
"""
import torch

def pairwise_squared_distances(A, B):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
