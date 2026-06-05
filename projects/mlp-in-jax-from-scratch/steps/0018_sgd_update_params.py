"""
Step 0018: sgd_update_params

Part 6 — Autodiff & SGD Update
Apply one SGD step: subtract lr * grad from each parameter leaf.
"""
import jax  # noqa: F401
import jax.numpy as jnp  # noqa: F401


def sgd_update_params(params, grads, lr):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
