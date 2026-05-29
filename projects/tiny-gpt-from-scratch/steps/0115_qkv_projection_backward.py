"""
Step 0115: qkv_projection_backward

Part 6 — Embeddings and Self-Attention
Backward of the Q/K/V projections: (dx, dWq, dWk, dWv).
"""
import numpy as np  # noqa: F401


def qkv_projection_backward(dq, dk, dv, x, wq, wk, wv):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
