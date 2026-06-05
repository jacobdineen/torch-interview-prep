"""
Step 0004: initial_vocab

Part 1 — Corpus & Symbols
Return the set of all distinct symbols appearing across the keys.
"""
import numpy as np  # noqa: F401


def initial_vocab(word_freqs):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
