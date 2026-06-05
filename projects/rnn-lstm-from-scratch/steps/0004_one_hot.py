"""
Step 0004: one_hot

Part 1 — Character Data
One-hot encode an integer id tensor along a new last axis of size vocab_size.
"""
import torch  # noqa: F401


def one_hot(ids, vocab_size):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
