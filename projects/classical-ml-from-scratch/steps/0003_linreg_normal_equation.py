"""
Step 0003: linreg_normal_equation

Part 1 — Linear Regression
Least-squares weights solving X w ~= y (X already includes any bias column).
"""
import numpy as np  # noqa: F401


def linreg_normal_equation(X, y):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
