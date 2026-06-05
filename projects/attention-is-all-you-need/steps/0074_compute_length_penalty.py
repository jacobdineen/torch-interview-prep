"""
Step 0074: compute_length_penalty

Part 11 — Decoding and Beam Search
Return the GNMT length-normalization factor for a hypothesis of the given length.
"""
import torch  # noqa: F401


def compute_length_penalty(length, alpha):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
