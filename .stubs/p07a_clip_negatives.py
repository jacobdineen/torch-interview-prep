"""
Problem 07a: clip_negatives replaces negatives with 0

(Split from parent problem 07: Problem 07: Boolean Masking and torch.where)
"""
import torch

def clip_negatives(x):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
