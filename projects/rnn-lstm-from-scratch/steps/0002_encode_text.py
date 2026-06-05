"""
Step 0002: encode_text

Part 1 — Character Data
Map each character in text to its integer id, returning a LongTensor (L,).
"""
import torch  # noqa: F401


def encode_text(text, stoi):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
