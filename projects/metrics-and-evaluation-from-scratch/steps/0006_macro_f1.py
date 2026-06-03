"""
Step 0006: macro_f1

Part 2 — Classification Metrics
Unweighted mean of per-class F1 from a confusion matrix (rows=true).
"""
import numpy as np  # noqa: F401


def macro_f1(cm):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
