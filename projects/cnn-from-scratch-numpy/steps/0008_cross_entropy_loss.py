"""
Step 0008: cross_entropy_loss

Part 1 — Softmax, Loss, and Metrics Primitives
Mean of -log(true-class prob + 1e-12) over the batch (scalar float).
"""
import numpy as np  # noqa: F401


def cross_entropy_loss(probs, labels):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
