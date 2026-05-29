"""
Step 0033: stable_softmax_2d_rowwise

Part 2 — NumPy and Softmax Foundations
Row-wise stable softmax of a 2-D array: each row sums to 1.
"""
import numpy as np  # noqa: F401


def stable_softmax_2d_rowwise(z):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
