"""
Problem 20c: smoothing=1.0 equals -mean(log_softmax)

(Split from parent problem 20: Problem 47: Label Smoothing Cross-Entropy)
"""
import torch
import torch.nn.functional as F

def label_smoothing_ce(logits, targets, smoothing=0.1, reduction='mean'):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
