"""
Step 0014: weighted_token_sum

Part 4 — Einsum — contractions
Weighted sum of value vectors: weights (B, Q, K) and values (B, K, D) give (B, Q, D).
"""
import numpy as np  # noqa: F401
from einops import rearrange, reduce, repeat, einsum, pack, unpack  # noqa: F401


def weighted_token_sum(weights, values):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
