"""
Step 0007: gather_true_class_probs

Part 1 — Softmax, Loss, and Metrics Primitives
Pick probs[i, labels[i]] for each row, shape (N,).
"""
import numpy as np  # noqa: F401


def gather_true_class_probs(probs, labels):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
