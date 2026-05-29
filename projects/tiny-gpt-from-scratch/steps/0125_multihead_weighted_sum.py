"""
Step 0125: multihead_weighted_sum

Part 6 — Embeddings and Self-Attention
Per-head weighted sum of values: weights @ V -> (B,H,T,d_head).
"""
import numpy as np  # noqa: F401


def multihead_weighted_sum(weights, v):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
