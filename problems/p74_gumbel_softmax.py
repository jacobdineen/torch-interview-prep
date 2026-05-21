"""
Problem 55: Gumbel-Softmax (with Straight-Through Estimator)

Differentiable approximation to categorical sampling.

  - gumbel_noise(shape, eps=1e-20, generator=None)
        Returns a tensor of shape `shape` with i.i.d. Gumbel(0, 1) samples computed as
        -log(-log(U)) where U ~ Uniform(0, 1).

  - gumbel_softmax(logits, tau=1.0, hard=False, generator=None)
        logits: (..., V)
        Compute y = softmax((logits + g) / tau) where g ~ Gumbel.
        If hard=True, return a one-hot using argmax via the straight-through estimator
        (forward: one-hot, backward: gradient flows through the soft y).
        Returns a tensor of the same shape as logits.
"""

import torch


def gumbel_noise(shape, eps=1e-20, generator=None):
    raise NotImplementedError


def gumbel_softmax(logits, tau=1.0, hard=False, generator=None):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
