"""
Problem 17d: huber_loss with delta=0.7 matches F.huber_loss

(Split from parent problem 17: Problem 17: MSE and Huber Loss From Scratch)
"""
import torch

def huber_loss(pred, target, delta=1.0, reduction='mean'):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
