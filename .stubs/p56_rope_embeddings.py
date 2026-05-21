"""
Problem 45: Rotary Positional Embedding (RoPE)

RoPE rotates pairs of dimensions in the query/key vectors as a function of position.
For dimension index pair (2i, 2i+1) and position p:

    theta_i = 1 / (base^(2i / d))
    angle   = p * theta_i
    (x_2i, x_2i+1) -> (x_2i * cos(angle) - x_2i+1 * sin(angle),
                      x_2i * sin(angle) + x_2i+1 * cos(angle))

Implement (assume d is even):

  - rope_freqs(seq_len, d, base=10000.0, device=None)
        Returns:
            cos: (seq_len, d/2)
            sin: (seq_len, d/2)

  - apply_rope(x, cos, sin)
        x: (..., T, d) -- typically (B, H, T, d)
        Rotates the last dim in pairs (2i, 2i+1).
        Returns a tensor of the same shape as x.

Property to satisfy in tests:
    <RoPE(q, pos=m), RoPE(k, pos=n)>  =  <RoPE(q, pos=m+s), RoPE(k, pos=n+s)>
i.e., dot-product depends only on relative position (m - n).
"""

import torch

def rope_freqs(seq_len, d, base=10000.0, device=None):
    raise NotImplementedError

def apply_rope(x, cos, sin):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
