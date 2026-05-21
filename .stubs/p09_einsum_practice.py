"""
Problem 09: einsum Practice

Implement each operation with a single torch.einsum call.

  - matmul(A, B)              : (M, K), (K, N)         -> (M, N)
  - batch_matmul(A, B)        : (B, M, K), (B, K, N)   -> (B, M, N)
  - bilinear(x, W, y)         : (B, D1), (D1, D2), (B, D2) -> (B,)  (x^T W y per batch)
  - attention_scores(q, k)    : (B, H, T, D), (B, H, T, D) -> (B, H, T, T)
                                (dot product over D for each (b, h, t_q, t_k))
"""

import torch

def matmul(A, B):
    raise NotImplementedError

def batch_matmul(A, B):
    raise NotImplementedError

def bilinear(x, W, y):
    raise NotImplementedError

def attention_scores(q, k):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
