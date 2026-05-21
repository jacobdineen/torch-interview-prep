"""
Problem 46: Multi-Head Attention With KV Cache

During autoregressive generation we don't want to re-encode the entire prefix on
every step. Implement an attention module that:
  * Encodes the new tokens' Q, K, V.
  * Concatenates new K and V to a cache.
  * Computes attention over the full cached K/V.

  - CachedMHA(d_model, num_heads)
        forward(x_new, cache=None):
            x_new: (B, T_new, d_model)  -- one or more new tokens
            cache: dict with keys "k", "v" (each (B, H, T_past, d_head)) or None.
            Returns:
                out:     (B, T_new, d_model)
                new_cache: dict with the updated k, v (concatenated along time).

Causal masking is handled inside (queries at step i must attend only to keys at
positions <= i within their absolute position in the cache).

Required correctness: streaming one token at a time must produce the same outputs as
running attention over the full sequence in one shot.
"""

import math
import torch
import torch.nn as nn

class CachedMHA(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        raise NotImplementedError

    def forward(self, x_new, cache=None):
        raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
