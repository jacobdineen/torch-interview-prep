"""
Problem 42: Multi-Head Attention (Standalone)

Implement multi-head self-attention as an nn.Module. Do NOT use nn.MultiheadAttention
or F.scaled_dot_product_attention.

  - MultiHeadAttention(d_model, num_heads, bias=True)
        * Has projection layers w_q, w_k, w_v, w_o, each nn.Linear(d_model, d_model, bias=bias).
        * forward(x, mask=None):
            x:    (B, T, d_model)
            mask: optional (B, T, T) or (1, 1, T, T) or (T, T) bool tensor. True = mask out.
            Returns (B, T, d_model).
        * Internally reshape to (B, H, T, d_head), do scaled dot-product attention,
          then concat heads and project with w_o.

Make sure that d_model % num_heads == 0.
"""

import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, bias=True):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, mask=None):
        raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
