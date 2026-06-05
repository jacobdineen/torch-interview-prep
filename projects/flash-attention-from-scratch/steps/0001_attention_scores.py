"""
Step 0001: attention_scores

Part 1 — Standard Attention
Compute the scaled dot-product attention score matrix S (N, M).
"""
import numpy as np  # noqa: F401


def attention_scores(Q, K, scale):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
