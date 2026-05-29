"""
Step 0128: merge_heads_to_d_model

Part 6 — Embeddings and Self-Attention
Concatenate heads back into d_model: (B,T,H,d_head) -> (B,T,d_model).
"""
import numpy as np  # noqa: F401


def merge_heads_to_d_model(x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
