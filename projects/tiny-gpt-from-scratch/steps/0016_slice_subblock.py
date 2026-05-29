"""
Step 0016: slice_subblock

Part 2 — NumPy and Softmax Foundations
Return the sub-block a[r0:r1, c0:c1].
"""
import numpy as np  # noqa: F401


def slice_subblock(a, r0, r1, c0, c1):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
