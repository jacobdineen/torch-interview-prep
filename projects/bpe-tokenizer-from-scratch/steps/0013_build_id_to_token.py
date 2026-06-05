"""
Step 0013: build_id_to_token

Part 4 — Tokenizing & Vocab
Return the inverse mapping from id back to token string.
"""
import numpy as np  # noqa: F401


def build_id_to_token(token_to_id):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
