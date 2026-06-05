"""
Step 0019: mask_attention_scores_with_neg_inf

Part 3 — Masks and Scaled Dot-Product Attention
Replace blocked attention scores with a large negative value before softmax.
"""
import torch  # noqa: F401


def mask_attention_scores_with_neg_inf(scores, mask):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
