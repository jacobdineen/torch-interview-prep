"""
Step 0024: sum_axis0

Part 2 — NumPy and Softmax Foundations
Sum down columns (over axis 0) -> shape (cols,).
"""
import numpy as np  # noqa: F401


def sum_axis0(a):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
