"""
Step 0018: train

Part 5 — Training
Run n_steps of full-batch SGD on the MLP and return the list of per-step loss floats.
"""
import math  # noqa: F401


def train(mlp, X, Y, lr, n_steps):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
