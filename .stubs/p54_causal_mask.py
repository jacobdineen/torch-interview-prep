"""
Problem 43: Causal Mask and Masked Attention

  - causal_mask(T, device=None)
        Returns a (T, T) bool tensor with True ABOVE the diagonal (positions that
        should be masked out -- i.e., future tokens).

  - apply_causal_mask(scores)
        scores: (..., T, T)
        Returns scores with the upper triangle (excluding the diagonal) set to -inf.

  - causal_attention(q, k, v)
        Causally-masked scaled dot-product attention. q, k, v: (B, H, T, D).
        Returns (B, H, T, D).

For the last function, the output at position t may only depend on q_t and on
k_0..k_t / v_0..v_t.
"""

import math
import torch

def causal_mask(T, device=None):
    raise NotImplementedError

def apply_causal_mask(scores):
    raise NotImplementedError

def causal_attention(q, k, v):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
