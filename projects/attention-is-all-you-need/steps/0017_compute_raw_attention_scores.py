"""
Step 0017: compute_raw_attention_scores

Part 3 — Masks and Scaled Dot-Product Attention
Compute unnormalized attention scores as the dot products between queries and keys.
"""
import torch  # noqa: F401


def compute_raw_attention_scores(Q, K):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
