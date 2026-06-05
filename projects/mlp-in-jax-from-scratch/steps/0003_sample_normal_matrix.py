"""
Step 0003: sample_normal_matrix

Part 1 — PRNG & Random Sampling
Sample a standard normal array of the given shape.
"""
import jax  # noqa: F401
import jax.numpy as jnp  # noqa: F401


def sample_normal_matrix(key, shape):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
