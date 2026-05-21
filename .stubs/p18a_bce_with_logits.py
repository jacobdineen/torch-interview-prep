"""
Problem 18a: bce_with_logits matches F.binary_cross_entropy_with_logits (reduction={...})

(Split from parent problem 18: Problem 19: Binary Cross-Entropy With Logits)
"""
import torch
import torch.nn.functional as F

def bce_with_logits(logits, targets, pos_weight=None, reduction='mean'):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
