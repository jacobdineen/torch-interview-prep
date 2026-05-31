"""
Step 0021: cross_entropy_loss

Part 3 — SFT Training Loop
Mean cross-entropy over all positions, ignoring ``ignore_index`` labels.
"""
import torch  # noqa: F401


def cross_entropy_loss(logits, labels, ignore_index=-100):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
