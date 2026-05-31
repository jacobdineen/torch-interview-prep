"""
Step 0043: value_loss_mse

Part 6 — Losses and Training Loop
Mean squared error between predicted and target values.
"""
import torch  # noqa: F401


def value_loss_mse(pred_values, target_values):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
