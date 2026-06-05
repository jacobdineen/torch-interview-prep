"""
Step 0002: stable_rowmax

Part 1 — Standard Attention
Return the per-row maximum of S over the last (keys) axis.
"""
import numpy as np  # noqa: F401


def stable_rowmax(S):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
