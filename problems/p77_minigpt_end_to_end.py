"""
Problem 77: MiniGPT End-to-End (Capstone)

Bring it all together: assemble a tiny GPT-style language model from the
components you built (token embedding, sinusoidal PE, GPT decoder blocks, tied
LM head) and train it on a synthetic copy-the-input task.

  - MiniGPT(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout=0.0)
        Components:
            tok_embed : nn.Embedding(vocab_size, d_model)
            pos_embed : LEARNED position embedding nn.Embedding(max_seq_len, d_model)
            blocks    : nn.ModuleList of GPT-style decoder blocks (causal self-attention + MLP)
            ln_f      : nn.LayerNorm(d_model)
            lm_head   : TIED with tok_embed (no separate parameter)
        forward(input_ids):
            input_ids: (B, T) long, T <= max_seq_len.
            Returns (B, T, vocab_size) logits.

  - causal_lm_loss(logits, targets, ignore_index=-100)
        Cross-entropy on the SHIFTED next-token prediction:
            logits  predict tokens at positions 1..T given tokens at positions 0..T-1.
        For an LM trained with input_ids == targets, the standard recipe is:
            loss = cross_entropy(logits[:, :-1].reshape(-1, V), targets[:, 1:].reshape(-1))

If you must restructure: keep the public functions the same.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout=0.0):
        super().__init__()
        raise NotImplementedError

    def forward(self, input_ids):
        raise NotImplementedError


def causal_lm_loss(logits, targets, ignore_index=-100):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
