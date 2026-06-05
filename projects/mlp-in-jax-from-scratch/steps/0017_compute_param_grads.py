"""
Step 0017: compute_param_grads

Part 6 — Autodiff & SGD Update
Gradient of the loss with respect to params (same pytree structure).
"""
import jax  # noqa: F401
import jax.numpy as jnp  # noqa: F401


def compute_param_grads(params, x, one_hot_targets):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
