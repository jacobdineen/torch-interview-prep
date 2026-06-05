"""
Step 0016: compute_sequence_loss

Part 4 — Loss & Training
Run forward_fn to get logits, then return their sequence cross-entropy.
"""
import torch  # noqa: F401


def compute_sequence_loss(params, X_onehot, targets, forward_fn):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
