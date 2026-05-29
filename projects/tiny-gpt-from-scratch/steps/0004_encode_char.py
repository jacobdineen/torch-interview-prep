"""
Step 0004: encode_char

Part 1 — Tokenizer
Encode a single character to its integer id.
"""
import numpy as np  # noqa: F401


def encode_char(ch, stoi):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
