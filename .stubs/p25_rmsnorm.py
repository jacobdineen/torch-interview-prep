"""
Problem 51: RMSNorm

Root Mean Square Layer Norm (LLaMA / T5). No mean subtraction; just normalize by RMS.

    y = x / sqrt(mean(x ** 2, dim=last_dims) + eps) * weight

  - RMSNorm(normalized_shape, eps=1e-6)
        * Parameter `weight` of shape normalized_shape, initialized to ones.
        * `normalized_shape` may be an int (normalize over last dim) or a tuple
          (normalize over the last len(shape) dims).
        * forward(x) returns a tensor of the same shape as x.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
