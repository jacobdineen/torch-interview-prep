"""
Step 0012: mlp_forward

Part 4 — Forward Pass Building Blocks
Run the MLP: linear+ReLU on every layer except the last, which is linear only.
"""
import jax  # noqa: F401
import jax.numpy as jnp  # noqa: F401


def mlp_forward(params, x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
