"""
Step 0041: sample_random_batch_offsets

Part 3 — Data Pipeline and Bigram Baseline
Sample ``batch_size`` random start offsets so each window fits in ``n``.
"""
import numpy as np  # noqa: F401


def sample_random_batch_offsets(n, block_size, batch_size, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
