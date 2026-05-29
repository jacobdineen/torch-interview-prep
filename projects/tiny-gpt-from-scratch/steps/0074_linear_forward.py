"""
Step 0074: linear_forward

Part 5 — Layer Primitives and Backprop
Linear layer without bias: y = x @ W.
"""
import numpy as np  # noqa: F401


def linear_forward(x, w):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
