"""
Problem 24: Embedding Layer

Implement nn.Embedding from scratch.

  - MyEmbedding(num_embeddings, embedding_dim, padding_idx=None)
        * Parameter `weight` of shape (num_embeddings, embedding_dim), initialized
          ~ N(0, 1).
        * If padding_idx is given, that row is zeroed and is not updated during training
          (i.e., the gradient at that row is zero).
        * forward(idx): idx is LongTensor of arbitrary shape; output has shape
          idx.shape + (embedding_dim,).

You can implement the lookup with simple advanced indexing.
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
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
