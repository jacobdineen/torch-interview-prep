"""
Step 0048: add_one_smoothing

Part 3 — Data Pipeline and Bigram Baseline
Laplace smoothing: add 1 to every count so no bigram has zero probability.
"""
import numpy as np  # noqa: F401


def add_one_smoothing(counts):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
