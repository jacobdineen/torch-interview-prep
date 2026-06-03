"""
Step 0007: roc_auc

Part 3 — Ranking & Regression Metrics
Binary ROC AUC via the rank (Mann-Whitney) formula, tie-safe. y in {0,1};
returns 0.5 if a class is absent.
"""
import numpy as np  # noqa: F401


def roc_auc(scores, y_true):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
