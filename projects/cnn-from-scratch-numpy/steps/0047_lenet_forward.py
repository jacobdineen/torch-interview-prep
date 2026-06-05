"""
Step 0047: lenet_forward

Part 5 — Assembling LeNet
Run conv blocks -> flatten -> FC blocks (ReLU) -> final linear logits; returns (logits, caches).
"""
import numpy as np  # noqa: F401


def lenet_forward(params, x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
