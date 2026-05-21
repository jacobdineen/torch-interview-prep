"""
Problem 61: GPT-Style Decoder Block

A single GPT decoder block stacks:

    h = x  + Drop( MHA( LN(x), causal_mask ) )       # pre-LN self-attention
    z = h  + Drop( MLP( LN(h) ) )

where MLP is Linear(d_model -> d_ff) -> GELU -> Dropout -> Linear(d_ff -> d_model).

Unlike the encoder block (Problem 59), the self-attention here is CAUSAL — token i
can only attend to tokens 0..i.

Implement:

  - GPTDecoderBlock(d_model, num_heads, d_ff, dropout=0.0)
        Components:
            ln1, ln2 : nn.LayerNorm(d_model)
            attn     : nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
            mlp      : nn.Sequential(Linear, GELU, Dropout, Linear)
            drop     : nn.Dropout(dropout)
        forward(x):
            x: (B, T, d_model)
            Returns (B, T, d_model).
            INTERNALLY generates a causal mask of shape (T, T) — caller does not pass one.

Modern LLMs typically swap LayerNorm for RMSNorm (Problem 25) and the GELU FFN for
SwiGLU (Problem 60), but this block is the canonical GPT-2 form.
"""

import torch
import torch.nn as nn


class GPTDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
