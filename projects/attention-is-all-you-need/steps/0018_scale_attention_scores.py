"""
Step 0018: scale_attention_scores

Part 3 — Masks and Scaled Dot-Product Attention
Scale attention scores down by the square root of the key dimension.
"""
import torch  # noqa: F401


def scale_attention_scores(scores, d_k):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
