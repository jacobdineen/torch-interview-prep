"""
Problem 16: Numerically Stable Softmax / LogSoftmax / LogSumExp

Implement (do NOT call torch's named softmax/log_softmax/logsumexp directly):

  - logsumexp(x, dim)
        Stable log-sum-exp along `dim`. Use the max-shift trick.

  - softmax(x, dim)
        Stable softmax: subtract per-axis max before exp.

  - log_softmax(x, dim)
        Equivalent to x - logsumexp(x, dim, keepdim=True).

Each must handle inputs containing very large positive numbers (e.g., 1e4) without
overflow.
"""

import torch

def logsumexp(x, dim):
    raise NotImplementedError

def softmax(x, dim):
    raise NotImplementedError

def log_softmax(x, dim):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
