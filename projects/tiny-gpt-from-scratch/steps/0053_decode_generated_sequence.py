"""
Step 0053: decode_generated_sequence

Part 3 — Data Pipeline and Bigram Baseline
Decode generated ids back to text.
"""
import numpy as np  # noqa: F401


def decode_generated_sequence(ids, itos):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
