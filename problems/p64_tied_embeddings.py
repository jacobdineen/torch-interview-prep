"""
Problem 64: Tied Input/Output Embeddings

In language models, the input token embedding matrix and the output (logit
projection) "lm head" are often tied — they share the same parameters. This halves
the parameter count for the largest matrix in many GPT-style models and improves
generalization.

Implement a tiny LM module that ties these weights:

  - TiedLM(vocab_size, d_model)
        Components:
            embed : nn.Embedding(vocab_size, d_model)
            (no separate lm_head Linear layer; share weights with `embed.weight`)
        forward(token_ids):
            token_ids: (B, T) long
            Returns (B, T, vocab_size) logits.
            Compute logits as hidden @ embed.weight.T (no bias).

The embed.weight tensor must be THE SAME OBJECT used both for the lookup and the
output projection (no copy). Test verifies this with `is` identity.
"""

import torch
import torch.nn as nn


class TiedLM(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        raise NotImplementedError

    def forward(self, token_ids):
        raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
