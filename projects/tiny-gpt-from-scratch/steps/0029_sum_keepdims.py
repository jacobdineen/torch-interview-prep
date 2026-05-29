"""
Step 0029: sum_keepdims

Part 2 — NumPy and Softmax Foundations
Sum over ``axis`` while keeping that dimension (size 1) for broadcasting.
"""
import numpy as np  # noqa: F401


def sum_keepdims(a, axis):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
