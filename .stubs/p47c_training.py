"""
Problem 47c: training loss decreases over 20 Adam steps

(Split from parent problem 47: Problem 36: Small CNN Classifier)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):

    def __init__(self, num_classes=10):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
