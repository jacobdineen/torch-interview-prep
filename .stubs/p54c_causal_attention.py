"""
Problem 54c: causal_attention output shape (B, H, T, D)

(Split from parent problem 54: Problem 43: Causal Mask and Masked Attention)
"""
import math
import torch

def causal_attention(q, k, v):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
