"""
Step 0049: backward_classifier_block

Part 5 — Assembling LeNet
Invert (ReLU ->) Linear; returns (dx, {'W':dW,'b':db}).
"""
import numpy as np  # noqa: F401


def backward_classifier_block(dout, cache):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
