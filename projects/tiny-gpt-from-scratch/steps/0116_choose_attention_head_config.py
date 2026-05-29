"""
Step 0116: choose_attention_head_config

Part 6 — Embeddings and Self-Attention
Per-head dimension d_head = d_model // n_heads (must divide evenly).
"""
import numpy as np  # noqa: F401


def choose_attention_head_config(d_model, n_heads):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
