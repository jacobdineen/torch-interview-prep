"""
Step 0124: multihead_masked_softmax_scores

Part 6 — Embeddings and Self-Attention
Per-head causal attention weights: softmax(QK^T/sqrt(d_head) + mask).
"""
import numpy as np  # noqa: F401


def multihead_masked_softmax_scores(q, k, mask):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
