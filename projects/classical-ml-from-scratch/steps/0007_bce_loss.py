"""
Step 0007: bce_loss

Part 2 — Logistic Regression
Mean binary cross-entropy; p are predicted probabilities in (0, 1).
"""
import numpy as np  # noqa: F401


def bce_loss(p, y):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
