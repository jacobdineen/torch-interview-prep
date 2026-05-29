"""
Step 0094: token_embedding_backward

Part 6 — Embeddings and Self-Attention
Gradient w.r.t. the table: scatter-add dout into the rows that were used.
"""
import numpy as np  # noqa: F401


def token_embedding_backward(dout, x, vocab_size, d_model):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
