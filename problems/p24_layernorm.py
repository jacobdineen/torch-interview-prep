"""
Problem 22: LayerNorm From Scratch

Implement nn.LayerNorm equivalent:

  - MyLayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)
        normalized_shape may be an int or tuple. Normalization happens over the LAST
        len(normalized_shape) dimensions, computing mean and population variance per
        instance (no running stats).
        If elementwise_affine: learnable weight (init ones) and bias (init zeros) of
        shape normalized_shape.

Match torch.nn.LayerNorm.
"""

import torch
import torch.nn as nn

class MyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
