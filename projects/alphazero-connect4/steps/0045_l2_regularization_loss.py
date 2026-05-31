"""
Step 0045: l2_regularization_loss

Part 6 — Losses and Training Loop
L2 penalty on all parameters.
"""
import torch  # noqa: F401


def l2_regularization_loss(net, weight_decay):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
