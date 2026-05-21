"""
Problem 30b: momentum + weight_decay matches torch.optim.SGD bit-for-bit

(Split from parent problem 30: Problem 26: SGD With Momentum From Scratch)
"""
import torch

class MySGD:

    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0):
        raise NotImplementedError

    @torch.no_grad()
    def step(self):
        raise NotImplementedError

    def zero_grad(self, set_to_none=True):
        raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
