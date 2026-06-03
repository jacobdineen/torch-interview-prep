"""
Step 0002: mse_loss

Part 1 — Linear Regression
Mean squared error between predictions and targets (a float).
"""
import numpy as np  # noqa: F401


def mse_loss(y_pred, y_true):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
