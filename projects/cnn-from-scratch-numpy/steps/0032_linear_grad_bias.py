"""
Step 0032: linear_grad_bias

Part 3 — Layer Forward and Backward Passes
Sum over rows; returns db.
"""
import numpy as np  # noqa: F401


def linear_grad_bias(dout):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
