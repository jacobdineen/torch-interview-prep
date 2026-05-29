"""
Step 0129: multihead_output_projection_forward

Part 6 — Embeddings and Self-Attention
Final output projection: (B,T,d_model) @ Wo.
"""
import numpy as np  # noqa: F401


def multihead_output_projection_forward(x, wo):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
