"""
Problem 44: Transformer Encoder Block (post-LN)

Implement a single Transformer encoder block:

    h = x + MHA(LN(x))                    # pre-LN; see below
    z = h + FFN(LN(h))

where FFN(u) = Linear(d_model -> d_ff), GELU, Dropout, Linear(d_ff -> d_model).
Use the **pre-LayerNorm** formulation (more stable, used by GPT and friends).

  - TransformerEncoderBlock(d_model, num_heads, d_ff, dropout=0.0)
        Components:
            self.ln1, self.ln2 : LayerNorm(d_model)
            self.attn          : MultiHeadAttention(d_model, num_heads)
              -- you may use nn.MultiheadAttention(batch_first=True), or import your
                 implementation from problem 42.
            self.ff            : nn.Sequential(Linear, GELU, Dropout, Linear)
            self.drop          : nn.Dropout(dropout) applied to each residual branch.
        forward(x, attn_mask=None):
            x: (B, T, d_model)
            attn_mask: bool (T, T) or None
            Returns: (B, T, d_model)
"""

import torch
import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, attn_mask=None):
        raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
