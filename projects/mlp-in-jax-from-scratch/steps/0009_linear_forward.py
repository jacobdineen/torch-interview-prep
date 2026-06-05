"""
Step 0009: linear_forward

Part 4 — Forward Pass Building Blocks
Apply an affine layer (W, b) to x: x @ W + b.
"""
import jax  # noqa: F401
import jax.numpy as jnp  # noqa: F401


def linear_forward(layer, x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
