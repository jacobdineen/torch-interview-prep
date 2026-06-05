"""
Step 0046: forward_classifier_block

Part 5 — Assembling LeNet
Linear (optionally followed by ReLU); returns (out, cache).
"""
import numpy as np  # noqa: F401


def forward_classifier_block(x, linear_layer, activation=True):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
