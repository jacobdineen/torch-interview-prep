"""
Step 0049: row_sums_of_counts

Part 3 — Data Pipeline and Bigram Baseline
Per-row totals, kept 2-D (vocab, 1) for broadcasting.
"""
import numpy as np  # noqa: F401


def row_sums_of_counts(counts):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
