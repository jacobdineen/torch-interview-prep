"""
Step 0120: transpose_heads_to_front

Part 6 — Embeddings and Self-Attention
(B,T,H,d_head) -> (B,H,T,d_head) so each head is an independent (T,d_head) block.
"""
import numpy as np  # noqa: F401


def transpose_heads_to_front(x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
