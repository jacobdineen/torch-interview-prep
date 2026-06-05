"""
Step 0013: mlp_forward

Part 4 — Neural Network
Thread the input list x through each layer in turn and return the last layer's outputs.
"""
import math  # noqa: F401


def mlp_forward(mlp, x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
