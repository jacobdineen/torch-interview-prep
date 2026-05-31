"""
Step 0049: training_step

Part 6 — Losses and Training Loop
One training step over a list of (state, policy, value) tuples. Returns loss.
"""
import torch  # noqa: F401


def training_step(net, batch, optimizer, weight_decay):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
