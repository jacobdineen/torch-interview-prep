"""
Problem 56d: apply_rope preserves L2 norm (it's a rotation)

(Split from parent problem 56: Problem 45: Rotary Positional Embedding (RoPE))
"""
import torch

def rope_freqs(seq_len, d, base=10000.0, device=None):
    raise NotImplementedError

def apply_rope(x, cos, sin):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
