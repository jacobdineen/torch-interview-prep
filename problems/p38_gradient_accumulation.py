"""
Problem 61: Gradient Accumulation

When the desired batch size doesn't fit in memory, accumulate gradients over
multiple micro-batches before stepping. The loss for each micro-batch must be
divided by `accum_steps` so the accumulated gradient matches the average gradient
over the effective batch.

  - train_step_with_accumulation(model, optimizer, micro_batches, loss_fn, accum_steps)
        * `micro_batches`: list of (x, y) pairs, length == accum_steps.
        * `loss_fn(model, x, y)` returns a scalar loss.
        * Zero grads once at the start, run forward+backward on each micro-batch,
          step ONCE at the end.
        * Each micro-batch's loss should be divided by accum_steps before backward.
        * Return the *total* loss (sum of the un-scaled per-batch losses).

The accumulated gradient should equal the gradient computed on the concatenated
big batch (when loss_fn returns mean MSE).
"""

import torch


def train_step_with_accumulation(model, optimizer, micro_batches, loss_fn, accum_steps):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
