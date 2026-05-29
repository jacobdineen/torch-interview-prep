"""
Step 0026: max_along_axis

Part 2 — NumPy and Softmax Foundations
Maximum over the given axis.
"""
import numpy as np  # noqa: F401


def max_along_axis(a, axis):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
