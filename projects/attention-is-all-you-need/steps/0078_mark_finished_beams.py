"""
Step 0078: mark_finished_beams

Part 11 — Decoding and Beam Search
Flag beams whose most recent token is the end-of-sequence symbol.
"""
import torch  # noqa: F401


def mark_finished_beams(sequences, eos_id=2):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
