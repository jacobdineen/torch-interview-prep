"""
Step 0014: cross_entropy_loss

Part 5 — Loss & Metrics
Compute the mean cross-entropy loss between logits and one-hot targets.
"""
import jax  # noqa: F401
import jax.numpy as jnp  # noqa: F401


def cross_entropy_loss(logits, one_hot_targets):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
