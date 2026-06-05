"""
Step 0008: apply_merge

Part 2 — Pair Statistics & Merging
Return a new word_freqs with pair merged in every key, preserving frequencies.
"""
import numpy as np  # noqa: F401


def apply_merge(word_freqs, pair):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
