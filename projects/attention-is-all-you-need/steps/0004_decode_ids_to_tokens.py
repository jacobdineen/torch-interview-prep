"""
Step 0004: decode_ids_to_tokens

Part 1 — Tokenization and Batching
Map each id in the sequence back to its token string, returning a list.
"""
import torch  # noqa: F401


def decode_ids_to_tokens(ids, id_to_token):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
