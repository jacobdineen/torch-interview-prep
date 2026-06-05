"""
Step 0027: flatten_forward

Part 3 — Layer Forward and Backward Passes
Reshape (N,C,H,W) -> (N, C*H*W); cache original shape; returns (out, cache).
"""
import numpy as np  # noqa: F401


def flatten_forward(x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
