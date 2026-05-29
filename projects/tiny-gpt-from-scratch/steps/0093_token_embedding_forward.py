"""
Step 0093: token_embedding_forward

Part 6 — Embeddings and Self-Attention
Look up embeddings for token ids x (B,T) -> (B, T, d_model).
"""
import numpy as np  # noqa: F401


def token_embedding_forward(tok_emb, x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
