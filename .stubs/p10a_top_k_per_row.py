"""
Problem 10a: top_k_per_row returns top-k values and indices, sorted desc

(Split from parent problem 10: Problem 10: Top-k, Sort, Argmax)
"""
import torch

def top_k_per_row(x, k):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
