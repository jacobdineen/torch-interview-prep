"""
Step 0071: run_one_training_step

Part 4 — Single-Layer Neural Bigram
Forward, loss, backward, and SGD update for one batch. Returns (W, loss).
"""
import numpy as np  # noqa: F401


def run_one_training_step(w, x, y, lr):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
