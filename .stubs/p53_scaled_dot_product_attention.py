"""
Problem 41: Scaled Dot-Product Attention

Implement the core attention operation without using torch.nn.functional.scaled_dot_product_attention.

  - scaled_dot_product_attention(q, k, v, mask=None)
        q: (..., Tq, D)
        k: (..., Tk, D)
        v: (..., Tk, Dv)
        mask: optional broadcastable bool tensor with shape (..., Tq, Tk).
              True means a position MUST BE MASKED OUT (set to -inf before softmax).
        Returns:
            out:     (..., Tq, Dv)
            attn:    (..., Tq, Tk) softmax weights (for inspection)

Steps:
    scores = q @ k.transpose(-2, -1) / sqrt(D)
    if mask is not None: scores = scores.masked_fill(mask, -inf)
    attn = softmax(scores, dim=-1)
    out  = attn @ v
"""

import math
import torch

def scaled_dot_product_attention(q, k, v, mask=None):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
