"""
Step 0010: fill_even_indices_with_sin

Part 2 — Embeddings and Positional Encoding
Write sine values into the even feature columns of the positional encoding table.
"""
import torch  # noqa: F401


def fill_even_indices_with_sin(pe, position, div_term):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
