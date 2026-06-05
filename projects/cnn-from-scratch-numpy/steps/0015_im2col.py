"""
Step 0015: im2col

Part 2 — Initialization and Convolution Plumbing
Extract sliding patches into (N, C*kh*kw, OH*OW), row=channel-major (c,ki,kj).
"""
import numpy as np  # noqa: F401


def im2col(x, kh, kw, stride, pad):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
