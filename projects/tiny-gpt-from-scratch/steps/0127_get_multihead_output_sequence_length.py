"""
Step 0127: get_multihead_output_sequence_length

Part 6 — Embeddings and Self-Attention
Sequence length T from a (B,T,H,d_head) tensor.
"""
import numpy as np  # noqa: F401


def get_multihead_output_sequence_length(x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
