"""
Step 0130: multihead_reshape_transpose_backward

Part 6 — Embeddings and Self-Attention
Inverse of reshape_to_heads + transpose_heads_to_front: take a per-head
gradient (B,H,T,d_head) back to (B,T,d_model).
"""
import numpy as np  # noqa: F401


def multihead_reshape_transpose_backward(dheads, n_heads):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
