"""
Step 0014: inertia

Part 3 — K-Means Clustering
Sum of squared distances from each point to its assigned centroid (a float).
"""
import numpy as np  # noqa: F401


def inertia(X, labels, C):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
