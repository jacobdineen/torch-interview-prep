"""
Step 0073: pick_next_token_by_argmax

Part 11 — Decoding and Beam Search
Greedily choose each position's next token as the highest-scoring vocab entry.
"""
import torch  # noqa: F401


def pick_next_token_by_argmax(logits):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
