"""
Problem 63: Sliding Window Attention

Mistral-style: each token attends to only the W most recent tokens (and itself),
on top of the usual causal restriction. This caps memory at O(T*W) rather than
O(T^2).

Implement:

  - sliding_window_causal_mask(T, window_size)
        Returns a (T, T) bool tensor where True positions are MASKED OUT.
        Position (i, j) is unmasked iff  (j <= i) and (i - j < window_size).

  - sliding_window_attention(q, k, v, window_size)
        q, k, v: (B, H, T, D).
        Returns (B, H, T, D) — scaled dot-product attention with the sliding-window
        causal mask applied (don't call F.scaled_dot_product_attention).
"""

import math
import torch


def sliding_window_causal_mask(T, window_size):
    raise NotImplementedError


def sliding_window_attention(q, k, v, window_size):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
