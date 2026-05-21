"""
Problem 64c: exactly one Parameter (the embedding) — lm_head is tied

(Split from parent problem 64: Problem 64: Tied Input/Output Embeddings)
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
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
