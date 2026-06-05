"""
Step 0020: train_mlp

Part 7 — Training Loop & Inference
Run n_steps of full-batch SGD, returning the final params and the list of per-step losses.
"""
import jax  # noqa: F401
import jax.numpy as jnp  # noqa: F401


def train_mlp(params, X, one_hot_targets, lr, n_steps):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
