"""
Problem 04a: center_rows subtracts per-row mean

(Split from parent problem 04: Problem 04: Broadcasting and Arithmetic)
"""
import torch

def center_rows(X):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
