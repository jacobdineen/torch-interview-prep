"""
Step 0003: row_sum

Part 1 — Softmax, Loss, and Metrics Primitives
Sum per row with keepdims, shape (N,1).
"""
import numpy as np  # noqa: F401


def row_sum(x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
