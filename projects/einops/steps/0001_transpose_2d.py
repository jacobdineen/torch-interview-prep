"""
Step 0001: transpose_2d

Part 1 — Rearrange — moving axes around
Transpose a 2-D array: swap its two axes.
"""
import numpy as np  # noqa: F401
from einops import rearrange, reduce, repeat, einsum, pack, unpack  # noqa: F401


def transpose_2d(x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
