"""
Step 0001: train_test_split

Part 1 — Splitting & Folds
Shuffle (seeded) and split into (X_train, X_test, y_train, y_test); the
test set is round(n * test_frac) rows.
"""
import numpy as np  # noqa: F401


def train_test_split(X, y, test_frac=0.25, seed=0):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
