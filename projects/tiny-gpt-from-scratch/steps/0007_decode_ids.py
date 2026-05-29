"""
Step 0007: decode_ids

Part 1 — Tokenizer
Decode a sequence of integer ids back into a string.
"""
import numpy as np  # noqa: F401


def decode_ids(ids, itos):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
