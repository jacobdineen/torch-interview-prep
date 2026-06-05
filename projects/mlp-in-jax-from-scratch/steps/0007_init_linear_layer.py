"""
Step 0007: init_linear_layer

Part 3 — Parameter Initialization
Initialize one linear layer: W = scale * normal((in_dim, out_dim)), b = zeros(out_dim).
"""
import jax  # noqa: F401
import jax.numpy as jnp  # noqa: F401


def init_linear_layer(key, in_dim, out_dim, scale=0.1):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
