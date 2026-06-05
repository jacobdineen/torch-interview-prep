"""
Step 0001: pretokenize

Part 1 — Corpus & Symbols
Split text on whitespace into a list of non-empty word strings.
"""
import numpy as np  # noqa: F401


def pretokenize(text):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
