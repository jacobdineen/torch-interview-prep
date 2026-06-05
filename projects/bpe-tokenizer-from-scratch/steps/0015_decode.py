"""
Step 0015: decode

Part 5 — Encode & Decode
Map ids back to tokens and reconstruct the original whitespace-joined text.
"""
import numpy as np  # noqa: F401


def decode(ids, id_to_token):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
