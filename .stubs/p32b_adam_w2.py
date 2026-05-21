"""
Problem 32b: AdamW (decoupled wd) differs from Adam (L2 wd) when wd > 0

(Split from parent problem 32: Problem 32: AdamW Optimizer (Decoupled Weight Decay))
"""
import torch

class MyAdamW:

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01):
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
