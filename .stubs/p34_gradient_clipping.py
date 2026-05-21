"""
Problem 29: Gradient Clipping by Global Norm

Implement clip-by-global-norm without calling torch.nn.utils.clip_grad_norm_.

  - clip_grad_norm_(params, max_norm, eps=1e-6)
        * params: iterable of nn.Parameter (only those with non-None grad are considered)
        * Compute total_norm = sqrt(sum_i ||g_i||_2^2) over flattened grads.
        * If total_norm > max_norm:
            scale = max_norm / (total_norm + eps)
            for each grad: g.mul_(scale)
        * Return total_norm (Tensor, scalar) -- the value BEFORE clipping.

Be careful: clipping is in-place on the existing .grad tensors.
"""

import torch
import torch.nn as nn

def clip_grad_norm_(params, max_norm, eps=1e-6):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
