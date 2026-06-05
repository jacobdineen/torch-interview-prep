"""
Step 0058: train_loop

Part 7 — Training Loop and Evaluation
Initialize optimizer state and train for n_epochs of Adam; return params and a loss history.
"""
import numpy as np  # noqa: F401


def train_loop(params, X, y, n_epochs, batch_size, lr, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
