"""
Step 0004: exp_shifted

Part 1 — Softmax, Loss, and Metrics Primitives
Numerically stable exp(x - row_max(x)), shape (N,K).
"""
import numpy as np  # noqa: F401


def exp_shifted(x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
