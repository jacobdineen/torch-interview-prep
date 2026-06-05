"""
Step 0014: build_padding_mask

Part 3 — Masks and Scaled Dot-Product Attention
Build a boolean mask marking real (non-pad) token positions for attention.
"""
import torch  # noqa: F401


def build_padding_mask(ids, pad_id=0):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
