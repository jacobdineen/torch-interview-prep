"""
Step 0002: k_fold_indices

Part 1 — Splitting & Folds
Return k (train_idx, test_idx) pairs of contiguous folds over range(n).
The test folds partition all n indices; train is the complement.
"""
import numpy as np  # noqa: F401


def k_fold_indices(n, k):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
