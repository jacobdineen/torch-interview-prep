"""
Problem 57c: symmetric bias shape (H, T, T)

(Split from parent problem 57: Problem 54: ALiBi Positional Bias)
"""
import torch

def alibi_slopes(num_heads):
    raise NotImplementedError

def alibi_bias(num_heads, seq_len, causal=True):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
