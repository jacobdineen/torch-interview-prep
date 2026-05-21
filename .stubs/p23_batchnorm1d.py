"""
Problem 21: BatchNorm1d From Scratch

Implement nn.BatchNorm1d-like behavior:

  - MyBatchNorm1d(num_features, momentum=0.1, eps=1e-5, affine=True)
        * Parameters (if affine=True): weight (gamma) and bias (beta), shape (num_features,)
        * Buffers: running_mean (init zeros) and running_var (init ones).
        * Forward expects (N, C).
        * Training mode:
            mean = batch.mean(dim=0)
            var  = batch.var(dim=0, unbiased=False)   # population var for normalization
            running_mean = (1 - momentum) * running_mean + momentum * mean
            running_var  = (1 - momentum) * running_var  + momentum * batch.var(dim=0, unbiased=True)
            (note: PyTorch uses unbiased var for the running stat update)
            out = (x - mean) / sqrt(var + eps)
        * Eval mode:
            out = (x - running_mean) / sqrt(running_var + eps)
        * If affine: out = out * weight + bias

Match torch.nn.BatchNorm1d when weights and buffers are equal.
"""

import torch
import torch.nn as nn

class MyBatchNorm1d(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5, affine=True):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
