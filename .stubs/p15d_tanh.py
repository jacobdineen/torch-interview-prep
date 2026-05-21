"""
Problem 15d: tanh matches torch.tanh

(Split from parent problem 15: Problem 15: Activation Functions From Scratch)
"""
import math
import torch

def tanh(x):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
