"""
Step 0027: evaluate_loss

Part 3 — SFT Training Loop
Mean SFT loss over a list of batches (no gradient).
"""
import torch  # noqa: F401


def evaluate_loss(model, batches):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
