"""
Step 0005: stable_softmax

Part 1 — Softmax, Loss, and Metrics Primitives
Row-wise softmax via exp_shifted / row_sum(exp_shifted), shape (N,K).
"""
import numpy as np  # noqa: F401


def stable_softmax(x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
