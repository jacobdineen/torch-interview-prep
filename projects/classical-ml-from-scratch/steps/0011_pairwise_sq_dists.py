"""
Step 0011: pairwise_sq_dists

Part 3 — K-Means Clustering
(n, k) squared Euclidean distances between rows of X (n,d) and C (k,d).
"""
import numpy as np  # noqa: F401


def pairwise_sq_dists(X, C):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
