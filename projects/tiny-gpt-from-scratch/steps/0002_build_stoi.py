"""
Step 0002: build_stoi

Part 1 — Tokenizer
Map each character to its index: {char: i} for the vocab list.
"""
import numpy as np  # noqa: F401


def build_stoi(vocab):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
