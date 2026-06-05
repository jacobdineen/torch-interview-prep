"""
Step 0007: scale_embeddings_by_sqrt_d_model

Part 2 — Embeddings and Positional Encoding
Scale token embeddings up by the square root of the model dimension.
"""
import torch  # noqa: F401


def scale_embeddings_by_sqrt_d_model(emb, d_model):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
