"""
Step 0033: linear_backward

Part 3 — Layer Forward and Backward Passes
Compose linear grads from cache; returns (dx, dW, db).
"""
import numpy as np  # noqa: F401


def linear_backward(dout, cache):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
