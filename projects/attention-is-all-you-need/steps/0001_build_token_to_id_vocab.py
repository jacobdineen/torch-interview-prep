"""
Step 0001: build_token_to_id_vocab

Part 1 — Tokenization and Batching
Build a token-to-id dict: the four special tokens first, then sorted unique whitespace tokens starting at id 4.
"""
import torch  # noqa: F401


def build_token_to_id_vocab(sentences):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
