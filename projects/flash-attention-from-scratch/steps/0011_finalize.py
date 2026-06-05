"""
Step 0011: finalize

Part 3 — Flash Forward (Tiled)
Normalize the unnormalized output accumulator by the running denominator.
"""
import numpy as np  # noqa: F401


def finalize(O_acc, l):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
