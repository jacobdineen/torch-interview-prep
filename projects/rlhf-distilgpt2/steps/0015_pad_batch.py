"""
Step 0015: pad_batch

Part 2 — SFT Data Pipeline
Right-pad a list of variable-length id sequences to a (batch, maxlen) tensor.
"""
import torch  # noqa: F401


def pad_batch(sequences, pad_value):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
