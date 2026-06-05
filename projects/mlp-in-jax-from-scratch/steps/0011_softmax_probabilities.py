"""
Step 0011: softmax_probabilities

Part 4 — Forward Pass Building Blocks
Numerically stable softmax over the last axis.
"""
import jax  # noqa: F401
import jax.numpy as jnp  # noqa: F401


def softmax_probabilities(logits):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
