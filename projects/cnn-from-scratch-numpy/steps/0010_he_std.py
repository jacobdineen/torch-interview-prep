"""
Step 0010: he_std

Part 2 — Initialization and Convolution Plumbing
He initialization standard deviation sqrt(2/fan_in).
"""
import numpy as np  # noqa: F401


def he_std(fan_in):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
