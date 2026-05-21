"""
Problem 03c: to_channels_last permutes (B, C, H, W) -> (B, H, W, C)

(Split from parent problem 03: Problem 03: Reshape, View, Permute)
"""
import torch

def to_channels_last(x):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
