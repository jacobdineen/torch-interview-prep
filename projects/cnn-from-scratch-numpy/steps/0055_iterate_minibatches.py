"""
Step 0055: iterate_minibatches

Part 6 — Synthetic Data Pipeline
Yield (Xb, yb) minibatches covering all data once, optionally shuffled.
"""
import numpy as np  # noqa: F401


def iterate_minibatches(X, y, batch_size, rng, shuffle=True):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
