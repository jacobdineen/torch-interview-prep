"""
Step 0082: relu_backward

Part 5 — Layer Primitives and Backprop
ReLU gradient: pass dout through only where the input was positive.
"""
import numpy as np  # noqa: F401


def relu_backward(dout, x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
