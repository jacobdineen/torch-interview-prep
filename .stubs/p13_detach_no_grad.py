"""
Problem 13: Controlling Gradient Flow

Implement two functions that demonstrate the difference between .detach() and
torch.no_grad(), plus a small utility.

  - param_norm(model):
        Return the L2 norm of all parameters in `model`, but the returned scalar must
        NOT participate in autograd (i.e., calling .backward() on something that uses
        it should not propagate gradients to the model). Use torch.no_grad().

  - stop_gradient_at(x, threshold):
        Return a tensor y such that y == x but for elements where x > threshold,
        gradients do not flow back through x. (Hint: torch.where with x.detach() vs x.)
        E.g., useful for "stop-gradient" tricks in self-supervised learning.

  - is_leaf_and_requires_grad(t):
        Return (is_leaf, requires_grad) as a tuple of bools.
"""

import torch
import torch.nn as nn

def param_norm(model):
    raise NotImplementedError

def stop_gradient_at(x, threshold):
    raise NotImplementedError

def is_leaf_and_requires_grad(t):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
