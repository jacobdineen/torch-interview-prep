"""
Problem 12: Custom Autograd Function

Implement a custom torch.autograd.Function for a "stable softplus" with a manual
backward pass. The forward is:

    y = softplus(x) = log(1 + exp(x))

Numerically stable form: y = max(x, 0) + log1p(exp(-|x|))

Backward:

    dL/dx = dL/dy * sigmoid(x)

Save what you need in ctx for the backward (you should NOT save y; you only need x).

Then provide:
  - stable_softplus(x): a Python wrapper calling YourFunction.apply.
"""

import torch
from torch.autograd import Function

class StableSoftplus(Function):
    @staticmethod
    def forward(ctx, x):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError

def stable_softplus(x):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
