"""
Problem 22b: matches F.triplet_margin_loss

(Split from parent problem 22: Problem 58: Triplet Margin Loss)
"""
import torch

def triplet_margin_loss(anchor, positive, negative, margin=1.0, reduction='mean'):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
