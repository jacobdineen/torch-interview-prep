"""
Step 0097: add_token_and_positional_embeddings

Part 6 — Embeddings and Self-Attention
Sum token (B,T,d) and positional (T,d) embeddings (pos broadcasts over batch).
"""
import numpy as np  # noqa: F401


def add_token_and_positional_embeddings(tok, pos):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
