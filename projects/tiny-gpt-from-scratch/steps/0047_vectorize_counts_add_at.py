"""
Step 0047: vectorize_counts_add_at

Part 3 — Data Pipeline and Bigram Baseline
Same tally, vectorized with np.add.at over all adjacent pairs.
"""
import numpy as np  # noqa: F401


def vectorize_counts_add_at(data, counts):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
