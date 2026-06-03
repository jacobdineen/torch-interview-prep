"""
Step 0005: precision_recall_f1

Part 2 — Classification Metrics
Binary precision, recall, F1 for the positive class (label 1), as a tuple.
Each is 0.0 when its denominator is zero.
"""
import numpy as np  # noqa: F401


def precision_recall_f1(y_true, y_pred):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
