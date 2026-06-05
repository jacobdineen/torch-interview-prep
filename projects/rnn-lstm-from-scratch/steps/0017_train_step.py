"""
Step 0017: train_step

Part 4 — Loss & Training
One SGD step: zero grads, backprop the sequence loss, update params in place, return the float loss.
"""
import torch  # noqa: F401


def train_step(params, X_onehot, targets, lr, forward_fn):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
