"""
Step 0029: linear_forward

Part 3 — Layer Forward and Backward Passes
x @ W + b; returns (out, cache).
"""
import numpy as np  # noqa: F401


def linear_forward(x, W, b):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
