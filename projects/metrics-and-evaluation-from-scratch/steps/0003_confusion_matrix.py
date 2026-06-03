"""
Step 0003: confusion_matrix

Part 2 — Classification Metrics
(num_classes, num_classes) integer matrix; rows are true, cols predicted.
"""
import numpy as np  # noqa: F401


def confusion_matrix(y_true, y_pred, num_classes):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
