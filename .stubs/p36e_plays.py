"""
Problem 36e: plays nicely with DataLoader

(Split from parent problem 36: Problem 32: Custom Collate for Variable-Length Sequences)
"""
import torch
from torch.utils.data import DataLoader

def pad_collate(batch, pad_value=0):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
