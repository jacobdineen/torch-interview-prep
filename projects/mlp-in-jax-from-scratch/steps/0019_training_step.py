"""
Step 0019: training_step

Part 7 — Training Loop & Inference
Perform one SGD step: compute the loss and grads, update params, return (new_params, loss).
"""
import jax  # noqa: F401
import jax.numpy as jnp  # noqa: F401


def training_step(params, x, one_hot_targets, lr):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
