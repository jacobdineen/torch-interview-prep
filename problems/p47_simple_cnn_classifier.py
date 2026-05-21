"""
Problem 36: Small CNN Classifier

Build a small CNN suitable for 28x28 grayscale images (MNIST-style) and train it for
a few steps on synthetic data to verify it learns.

  - SmallCNN(num_classes=10)
        Block 1: Conv2d(1 -> 16, k=3, p=1), ReLU, MaxPool2d(2)        # 28 -> 14
        Block 2: Conv2d(16 -> 32, k=3, p=1), ReLU, MaxPool2d(2)       # 14 -> 7
        Head:    flatten, Linear(32*7*7 -> 64), ReLU, Linear(64 -> num_classes)

  - count_parameters(model): total number of trainable parameters as an int.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

def count_parameters(model):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
