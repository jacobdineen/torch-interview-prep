"""
Step 0002: word_to_symbols

Part 1 — Corpus & Symbols
Return the word's characters followed by the END marker as a tuple.
"""
import numpy as np  # noqa: F401


def word_to_symbols(word):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
