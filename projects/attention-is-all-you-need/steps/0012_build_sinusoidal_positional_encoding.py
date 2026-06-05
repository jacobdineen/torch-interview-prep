"""
Step 0012: build_sinusoidal_positional_encoding

Part 2 — Embeddings and Positional Encoding
Assemble the full fixed sinusoidal positional encoding table of shape (max_len, d_model).
"""
import torch  # noqa: F401


def build_sinusoidal_positional_encoding(max_len, d_model):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
