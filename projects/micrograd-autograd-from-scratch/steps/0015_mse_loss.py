"""
Step 0015: mse_loss

Part 5 — Training
Return the summed squared-error Value between predictions and target floats.
"""
import math  # noqa: F401


def mse_loss(preds, targets):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
