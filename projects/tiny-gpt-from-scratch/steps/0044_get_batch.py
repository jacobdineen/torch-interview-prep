"""
Step 0044: get_batch

Part 3 — Data Pipeline and Bigram Baseline
Sample a random (X, Y) batch of (batch_size, block_size) token windows.
"""
import numpy as np  # noqa: F401


def get_batch(data, block_size, batch_size, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
