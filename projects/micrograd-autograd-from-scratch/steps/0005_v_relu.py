"""
Step 0005: v_relu

Part 2 — Activations
Apply the rectified linear unit (clamp negatives to zero) to a Value.
"""
import math  # noqa: F401


def v_relu(a):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
