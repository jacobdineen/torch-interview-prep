"""
Problem 15: Activation Functions From Scratch

Implement these elementwise activations without using torch.nn.functional's named
versions. You may use torch.{exp, tanh, erf, where, clamp, ...}.

  - relu(x)
  - leaky_relu(x, negative_slope=0.01)
  - sigmoid(x)            # use exp; consider numerical stability for large |x|
  - tanh(x)               # via exp; don't use torch.tanh
  - gelu(x)               # exact (erf-based): x * 0.5 * (1 + erf(x / sqrt(2)))
"""

import math
import torch

def relu(x):
    raise NotImplementedError

def leaky_relu(x, negative_slope=0.01):
    raise NotImplementedError

def sigmoid(x):
    raise NotImplementedError

def tanh(x):
    raise NotImplementedError

def gelu(x):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
