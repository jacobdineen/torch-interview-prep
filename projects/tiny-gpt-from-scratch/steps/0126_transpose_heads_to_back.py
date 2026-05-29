"""
Step 0126: transpose_heads_to_back

Part 6 — Embeddings and Self-Attention
(B,H,T,d_head) -> (B,T,H,d_head), ready to merge heads.
"""
import numpy as np  # noqa: F401


def transpose_heads_to_back(x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
