"""
Step 0001: make_prng_key

Part 1 — PRNG & Random Sampling
Create a PRNG key from an integer seed.
"""
import jax  # noqa: F401
import jax.numpy as jnp  # noqa: F401


def make_prng_key(seed):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
