"""
Step 0014: encode

Part 5 — Encode & Decode
Tokenize text into BPE tokens and map them to their integer ids.
"""
import numpy as np  # noqa: F401


def encode(text, merge_ranks, token_to_id):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
