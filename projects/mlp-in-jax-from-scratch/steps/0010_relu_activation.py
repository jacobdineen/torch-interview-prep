"""
Step 0010: relu_activation

Part 4 — Forward Pass Building Blocks
Elementwise ReLU: max(x, 0).
"""
import jax  # noqa: F401
import jax.numpy as jnp  # noqa: F401


def relu_activation(x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
