"""
Problem 31b: Adam converges on a quadratic

(Split from parent problem 31: Problem 27: Adam Optimizer From Scratch)
"""

import torch


class MyAdam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        raise NotImplementedError

    @torch.no_grad()
    def step(self):
        raise NotImplementedError

    def zero_grad(self, set_to_none=True):
        raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
