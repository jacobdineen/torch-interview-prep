"""
Step 0002: build_id_to_token_vocab

Part 1 — Tokenization and Batching
Return the inverse mapping from id to token.
"""
import torch  # noqa: F401


def build_id_to_token_vocab(token_to_id):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
