"""
Step 0012: init_zero_bias

Part 2 — Initialization and Convolution Plumbing
Return a length-n zero bias vector.
"""
import numpy as np  # noqa: F401


def init_zero_bias(n):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
