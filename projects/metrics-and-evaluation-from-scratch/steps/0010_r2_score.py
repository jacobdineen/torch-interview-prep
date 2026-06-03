"""
Step 0010: r2_score

Part 3 — Ranking & Regression Metrics
Coefficient of determination R^2 = 1 - SS_res / SS_tot (a float).
"""
import numpy as np  # noqa: F401


def r2_score(y_true, y_pred):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
