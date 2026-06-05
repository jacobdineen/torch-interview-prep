"""
Step 0018: train_model

Part 4 — Loss & Training
Run n_steps of train_step and return the list of per-step losses.
"""
import torch  # noqa: F401


def train_model(params, X_onehot, targets, lr, n_steps, forward_fn):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
