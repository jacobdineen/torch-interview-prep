"""
Step 0104: scale_attention_scores

Part 6 — Embeddings and Self-Attention
Scale scores by 1/sqrt(d_head) to stabilize the softmax.
"""
import numpy as np  # noqa: F401


def scale_attention_scores(scores, d_head):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
