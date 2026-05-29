"""
Step 0114: qk_scores_backward

Part 6 — Embeddings and Self-Attention
Backward of scores = Q @ K.T: (dQ, dK).
"""
import numpy as np  # noqa: F401


def qk_scores_backward(dscores, q, k):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
