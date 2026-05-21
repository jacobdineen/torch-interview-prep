"""
Problem 28a: weight has shape (V, D)

(Split from parent problem 28: Problem 24: Embedding Layer)
"""

import torch
import torch.nn as nn


class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        raise NotImplementedError

    def forward(self, idx):
        raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
