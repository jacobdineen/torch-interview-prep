"""
Step 0119: reshape_to_heads

Part 6 — Embeddings and Self-Attention
Split the feature axis into heads: (B,T,d_model) -> (B,T,n_heads,d_head).
"""
import numpy as np  # noqa: F401


def reshape_to_heads(x, n_heads):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
