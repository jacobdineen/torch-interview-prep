"""
Step 0076: select_top_k_candidates

Part 11 — Decoding and Beam Search
Pick the k best (beam, token) candidates across the flattened score grid.
"""
import torch  # noqa: F401


def select_top_k_candidates(scores, k):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
