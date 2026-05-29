"""
Step 0030: naive_softmax_1d

Part 2 — NumPy and Softmax Foundations
Softmax of a 1-D vector, the textbook (overflow-prone) way: exp / sum(exp).
"""
import numpy as np  # noqa: F401


def naive_softmax_1d(z):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
