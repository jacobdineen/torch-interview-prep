"""
Step 0046: combined_loss

Part 6 — Losses and Training Loop
AlphaZero loss: value MSE + policy cross-entropy + L2.
"""
import torch  # noqa: F401


def combined_loss(value_loss, policy_loss, l2_loss):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
