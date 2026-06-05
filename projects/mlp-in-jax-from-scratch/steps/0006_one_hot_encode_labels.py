"""
Step 0006: one_hot_encode_labels

Part 2 — Synthetic Data & Labels
Return float one-hot rows of the integer labels with n_classes columns.
"""
import jax  # noqa: F401
import jax.numpy as jnp  # noqa: F401


def one_hot_encode_labels(labels, n_classes):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
