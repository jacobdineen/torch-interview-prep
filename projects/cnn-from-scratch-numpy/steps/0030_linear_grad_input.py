"""
Step 0030: linear_grad_input

Part 3 — Layer Forward and Backward Passes
dout @ W.T; returns dx.
"""
import numpy as np  # noqa: F401


def linear_grad_input(dout, W):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
