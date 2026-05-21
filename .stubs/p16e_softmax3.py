"""
Problem 16e: softmax stable at logits of order 1e4

(Split from parent problem 16: Problem 16: Numerically Stable Softmax / LogSoftmax / LogSumExp)
"""
import torch

def softmax(x, dim):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
