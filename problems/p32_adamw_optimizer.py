"""
Problem 32: AdamW Optimizer (Decoupled Weight Decay)

AdamW differs from Adam in how weight decay is applied. Adam folds weight decay
into the gradient (L2 regularization), but that interacts with the Adam moments.
AdamW instead applies decay DIRECTLY to the parameter, AFTER the Adam update:

    m = beta1 * m + (1 - beta1) * g            # g is RAW gradient (no L2 term)
    v = beta2 * v + (1 - beta2) * g * g
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    p   -= lr * m_hat / (sqrt(v_hat) + eps)
    p   -= lr * weight_decay * p               # decoupled decay

This is the version used to train basically every modern transformer.

Implement:
  - MyAdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)

Must match torch.optim.AdamW exactly over multiple steps.
"""

import torch


class MyAdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
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
