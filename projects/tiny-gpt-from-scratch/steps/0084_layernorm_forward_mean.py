"""
Step 0084: layernorm_forward_mean

Part 5 — Layer Primitives and Backprop
Per-row mean over the feature axis, kept 2-D for broadcasting.
"""
import numpy as np  # noqa: F401


def layernorm_forward_mean(x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
