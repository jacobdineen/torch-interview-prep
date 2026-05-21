"""
Problem 45b: stride=None defaults to kernel_size

(Split from parent problem 45: Problem 34: MaxPool2d From Scratch)
"""
import torch
import torch.nn.functional as F

def my_max_pool2d(x, kernel_size, stride=None, padding=0):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
