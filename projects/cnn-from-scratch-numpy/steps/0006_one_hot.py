"""
Step 0006: one_hot

Part 1 — Softmax, Loss, and Metrics Primitives
One-hot encode integer labels into float rows, shape (N, num_classes).
"""
import numpy as np  # noqa: F401


def one_hot(labels, num_classes):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
