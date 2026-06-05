"""
Step 0003: build_word_frequencies

Part 1 — Corpus & Symbols
Map each word's symbol-tuple to the number of times that word occurs.
"""
import numpy as np  # noqa: F401


def build_word_frequencies(words):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
