"""
Problem 19a: cross_entropy matches F.cross_entropy (reduction={...})

(Split from parent problem 19: Problem 18: Cross-Entropy From Scratch)
"""
import torch
import torch.nn.functional as F

def cross_entropy(logits, targets, reduction='mean', ignore_index=None):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
