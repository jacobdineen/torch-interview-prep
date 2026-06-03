"""
Step 0012: assign_clusters

Part 3 — K-Means Clustering
Index of the nearest centroid for each row of X.
"""
import numpy as np  # noqa: F401


def assign_clusters(X, C):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
