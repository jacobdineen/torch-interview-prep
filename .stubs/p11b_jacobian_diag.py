"""
Problem 11b: jacobian_diag of f(z)=z^3 equals 3*z^2

(Split from parent problem 11: Problem 11: Autograd Basics)
"""
import torch

def jacobian_diag(f, x):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
