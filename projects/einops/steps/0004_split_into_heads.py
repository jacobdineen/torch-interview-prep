"""
Step 0004: split_into_heads

Part 1 — Rearrange — moving axes around
Reshape (batch, seq, dim) into (batch, heads, seq, head_dim) by splitting the last axis into `heads` equal groups.
"""
import numpy as np  # noqa: F401
from einops import rearrange, reduce, repeat, einsum, pack, unpack  # noqa: F401


def split_into_heads(x, heads):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
