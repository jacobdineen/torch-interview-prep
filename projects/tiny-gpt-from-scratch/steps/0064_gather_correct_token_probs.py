"""
Step 0064: gather_correct_token_probs

Part 4 — Single-Layer Neural Bigram
Pick out the probability assigned to each true next token y. Shape (B,).
"""
import numpy as np  # noqa: F401


def gather_correct_token_probs(probs, y):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
