"""
Step 0028: transpose_matrix

Part 2 — NumPy and Softmax Foundations
Transpose of a 2-D array.
"""
import numpy as np  # noqa: F401


def transpose_matrix(a):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
