"""
Step 0053: shuffle_indices

Part 6 — Synthetic Data Pipeline
Return a random permutation of range(n) using rng.
"""
import numpy as np  # noqa: F401


def shuffle_indices(n, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
