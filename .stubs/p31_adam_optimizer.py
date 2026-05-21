"""
Problem 27: Adam Optimizer From Scratch

Implement Adam matching torch.optim.Adam (no AMSGrad, no decoupled weight decay).

  - MyAdam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        For each parameter p with gradient g:
            if weight_decay: g = g + weight_decay * p           # L2 form, as in Adam
            step += 1
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g * g
            m_hat = m / (1 - beta1 ** step)
            v_hat = v / (1 - beta2 ** step)
            p.data -= lr * m_hat / (sqrt(v_hat) + eps)

Must compare bit-tight against torch.optim.Adam over multiple steps.
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
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
