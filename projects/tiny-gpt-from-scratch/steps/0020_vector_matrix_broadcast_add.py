"""
Step 0020: vector_matrix_broadcast_add

Part 2 — NumPy and Softmax Foundations
Add a length-``cols`` vector to each row of matrix ``m`` (broadcasting).
"""
import numpy as np  # noqa: F401


def vector_matrix_broadcast_add(m, v):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
