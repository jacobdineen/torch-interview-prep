"""
Step 0032: stable_softmax_1d

Part 2 — NumPy and Softmax Foundations
Numerically stable softmax: subtract the max before exponentiating.
"""
import numpy as np  # noqa: F401


def stable_softmax_1d(z):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
