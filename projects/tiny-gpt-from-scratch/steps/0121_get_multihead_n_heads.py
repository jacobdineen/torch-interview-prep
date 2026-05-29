"""
Step 0121: get_multihead_n_heads

Part 6 — Embeddings and Self-Attention
Number of heads from a (B,H,T,d_head) tensor.
"""
import numpy as np  # noqa: F401


def get_multihead_n_heads(x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
