"""
Step 0016: covariance_matrix

Part 4 — PCA
Sample covariance of already-centered data: Xc^T Xc / (n - 1).
"""
import numpy as np  # noqa: F401


def covariance_matrix(Xc):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
