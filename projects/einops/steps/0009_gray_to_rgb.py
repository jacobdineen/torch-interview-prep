"""
Step 0009: gray_to_rgb

Part 3 — Repeat — broadcasting and tiling
Turn a 2-D grayscale image (H, W) into an (H, W, 3) RGB image by copying it across 3 channels.
"""
import numpy as np  # noqa: F401
from einops import rearrange, reduce, repeat, einsum, pack, unpack  # noqa: F401


def gray_to_rgb(x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
