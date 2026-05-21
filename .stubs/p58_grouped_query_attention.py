"""
Problem 58: Grouped-Query / Multi-Query Attention

Modern LLMs (LLaMA-2/3, Mistral, ...) reduce KV-cache memory by giving K and V
fewer heads than Q. Each query head is assigned to a "kv group"; all query heads
in the same group share K and V.

    num_q_heads     = H
    num_kv_groups   = G       (must divide H)
    head_dim        = D

  - GroupedQueryAttention(d_model, num_q_heads, num_kv_groups, bias=False)
        * w_q: Linear(d_model, H*D)
        * w_k: Linear(d_model, G*D)
        * w_v: Linear(d_model, G*D)
        * w_o: Linear(H*D, d_model)
        * forward(x, mask=None):
            - Reshape Q to (B, H, T, D)
            - Reshape K, V to (B, G, T, D), then repeat_interleave along the group
              axis by H // G so they become (B, H, T, D) effectively.
            - Standard scaled dot-product attention, mask is broadcast-compatible.

When G == H this degrades to vanilla MHA. When G == 1 it is multi-query attention.
"""

import math
import torch
import torch.nn as nn


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_q_heads, num_kv_groups, bias=False):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, mask=None):
        raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
