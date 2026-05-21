"""
Problem 50: InfoNCE / SimCLR-Style Contrastive Loss

Given two views of N samples, compute the symmetric InfoNCE loss used in SimCLR /
CLIP. For two batches of embeddings:

    a_i, b_i   for i in [0, N)

the loss is the average of two cross-entropies:

    L_ab  = - mean_i log[ exp(sim(a_i, b_i) / tau)  /  sum_j exp(sim(a_i, b_j) / tau) ]
    L_ba  = - mean_i log[ exp(sim(b_i, a_i) / tau)  /  sum_j exp(sim(b_i, a_j) / tau) ]
    L     = (L_ab + L_ba) / 2

with sim(u, v) being cosine similarity (l2-normalize first), and tau the temperature.

Implement:
  - info_nce_loss(a, b, tau=0.1)
        a, b: (N, D) embeddings (will be normalized inside the function).
        Returns a 0-D Tensor loss.

Property to check: when a and b are equal AND well-separated rows (e.g.,
non-collinear), the loss should be small. When a and b are uncorrelated, the loss
should be close to log(N).
"""

import math
import torch
import torch.nn.functional as F

def info_nce_loss(a, b, tau=0.1):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
