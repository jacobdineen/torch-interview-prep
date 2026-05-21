"""
Problem 26: SGD With Momentum From Scratch

Implement a minimal optimizer with the same interface as torch.optim.SGD.

  - MySGD(params, lr, momentum=0.0, weight_decay=0.0)
        * step(): for each param p with grad g:
            if weight_decay: g = g + weight_decay * p
            if momentum:
                buf = momentum * buf + g            (PyTorch default; not Nesterov)
                g = buf
            p.data -= lr * g
        * zero_grad(set_to_none=True)

You may NOT call torch.optim.* anywhere in your implementation.
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



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
