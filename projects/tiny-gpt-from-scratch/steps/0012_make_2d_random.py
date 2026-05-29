"""
Step 0012: make_2d_random

Part 2 — NumPy and Softmax Foundations
Return a (rows, cols) array of standard-normal samples drawn from ``rng``.
"""
import numpy as np  # noqa: F401


def make_2d_random(rows, cols, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
