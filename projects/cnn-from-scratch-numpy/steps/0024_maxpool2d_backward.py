"""
Step 0024: maxpool2d_backward

Part 3 — Layer Forward and Backward Passes
Scatter gradients back to the stored max positions; returns dx.
"""
import numpy as np  # noqa: F401


def maxpool2d_backward(dout, cache):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
