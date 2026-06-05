"""
Step 0005: count_pairs

Part 2 — Pair Statistics & Merging
Count every adjacent symbol pair, weighted by each word's frequency.
"""
import numpy as np  # noqa: F401


def count_pairs(word_freqs):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
