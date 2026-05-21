"""
Problem 10b: sort_then_take_indices returns k smallest by index (ascending order of value)

(Split from parent problem 10: Problem 10: Top-k, Sort, Argmax)
"""
import torch

def sort_then_take_indices(x, k):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
