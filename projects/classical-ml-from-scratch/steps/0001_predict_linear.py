"""
Step 0001: predict_linear

Part 1 — Linear Regression
Linear model prediction: X @ w + b.
"""
import numpy as np  # noqa: F401


def predict_linear(X, w, b):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
