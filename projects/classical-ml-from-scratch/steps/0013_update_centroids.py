"""
Step 0013: update_centroids

Part 3 — K-Means Clustering
New centroids = mean of the points assigned to each of k clusters
(an empty cluster keeps a zero row).
"""
import numpy as np  # noqa: F401


def update_centroids(X, labels, k):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
