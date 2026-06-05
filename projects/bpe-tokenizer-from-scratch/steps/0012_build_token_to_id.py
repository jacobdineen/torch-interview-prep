"""
Step 0012: build_token_to_id

Part 4 — Tokenizing & Vocab
Assign ids to the sorted initial symbols, then to each merged token in merge order, with no duplicates.
"""
import numpy as np  # noqa: F401


def build_token_to_id(initial_symbols, merges):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
