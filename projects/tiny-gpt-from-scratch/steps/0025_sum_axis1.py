"""
Step 0025: sum_axis1

Part 2 — NumPy and Softmax Foundations
Sum across rows (over axis 1) -> shape (rows,).
"""
import numpy as np  # noqa: F401


def sum_axis1(a):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
