"""
Step 0024: transpose_heads_before_sequence

Part 4 — Multi-Head Attention
Move the head dimension ahead of the sequence dimension.
"""
import torch  # noqa: F401


def transpose_heads_before_sequence(x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
