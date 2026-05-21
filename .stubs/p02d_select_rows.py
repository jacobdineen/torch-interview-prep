"""
Problem 02d: select_rows picks rows by index

(Split from parent problem 02: Problem 02: Indexing and Slicing)
"""
import torch

def select_rows(x, idx):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
