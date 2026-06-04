"""
Step 0013: attention_scores

Part 4 — Einsum — contractions
Dot-product attention scores: queries (B, Q, D) and keys (B, K, D) give (B, Q, K).
"""
import numpy as np  # noqa: F401
from einops import rearrange, reduce, repeat, einsum, pack, unpack  # noqa: F401


def attention_scores(q, k):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
