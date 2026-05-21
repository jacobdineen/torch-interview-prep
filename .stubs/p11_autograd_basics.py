"""
Problem 11: Autograd Basics

Implement functions that exercise the autograd API.

  - grad_of_sum_of_squares(x):
        Given x with requires_grad=False, return the gradient of
        L(x) = sum(x^2) with respect to x using autograd. Returns a Tensor.

  - jacobian_diag(f, x):
        f maps a 1D tensor of length N to a 1D tensor of length N (elementwise).
        Return the diagonal of the Jacobian df/dx (length N) using autograd.
        Hint: use torch.autograd.grad with grad_outputs.

  - second_derivative(x):
        L(x) = sum(x^4). Return the diagonal of the Hessian (= 12 * x^2) using
        create_graph and a second autograd call.
"""

import torch

def grad_of_sum_of_squares(x):
    raise NotImplementedError

def jacobian_diag(f, x):
    raise NotImplementedError

def second_derivative(x):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
