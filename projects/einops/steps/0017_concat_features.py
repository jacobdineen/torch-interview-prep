"""
Step 0017: concat_features

Part 5 — Pack & unpack — variadic axes
Concatenate two feature maps (H, W, C1) and (H, W, C2) along the channel axis, giving (H, W, C1+C2).
"""
import numpy as np  # noqa: F401
from einops import rearrange, reduce, repeat, einsum, pack, unpack  # noqa: F401


def concat_features(a, b):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
