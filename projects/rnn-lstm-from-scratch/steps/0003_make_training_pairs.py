"""
Step 0003: make_training_pairs

Part 1 — Character Data
Split ids into non-overlapping windows X and next-char targets Y, dropping the remainder.
"""
import torch  # noqa: F401


def make_training_pairs(ids, seq_len):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
