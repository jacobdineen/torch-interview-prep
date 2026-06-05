"""
Step 0015: build_causal_mask

Part 3 — Masks and Scaled Dot-Product Attention
Build a lower-triangular boolean mask allowing each position to attend only to earlier or equal positions.
"""
import torch  # noqa: F401


def build_causal_mask(seq_len):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
