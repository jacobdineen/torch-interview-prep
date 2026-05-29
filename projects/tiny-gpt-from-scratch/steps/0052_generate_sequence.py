"""
Step 0052: generate_sequence

Part 3 — Data Pipeline and Bigram Baseline
Generate ``n`` tokens from the bigram table, starting at ``start_token``.
"""
import numpy as np  # noqa: F401


def generate_sequence(probs, start_token, n, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
