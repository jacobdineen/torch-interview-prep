"""
Step 0099: create_qkv_projections

Part 6 — Embeddings and Self-Attention
Single-head Q/K/V projection matrices, each (d_model, d_model).
"""
import numpy as np  # noqa: F401


def create_qkv_projections(d_model):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
