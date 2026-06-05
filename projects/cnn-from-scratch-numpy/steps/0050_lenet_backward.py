"""
Step 0050: lenet_backward

Part 5 — Assembling LeNet
Backprop through the whole net; returns grads with the same structure as params.
"""
import numpy as np  # noqa: F401


def lenet_backward(dlogits, caches):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
