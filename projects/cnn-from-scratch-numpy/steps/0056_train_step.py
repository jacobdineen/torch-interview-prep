"""
Step 0056: train_step

Part 7 — Training Loop and Evaluation
One optimization step: lenet forward, softmax-CE loss/grad, lenet backward, Adam-update every param.
"""
import numpy as np  # noqa: F401


def train_step(params, X, y, lr, opt_state, t):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
