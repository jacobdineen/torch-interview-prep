"""
Step 0105: build_causal_mask

Part 6 — Embeddings and Self-Attention
Boolean (T,T) mask, True where a position must be hidden (strictly future).
"""
import numpy as np  # noqa: F401


def build_causal_mask(t):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
