"""
Step 0003: softmax_rows

Part 1 — Standard Attention
Compute a numerically stable row-wise softmax of S.
"""
import numpy as np  # noqa: F401


def softmax_rows(S):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
