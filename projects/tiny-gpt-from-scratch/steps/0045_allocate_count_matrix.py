"""
Step 0045: allocate_count_matrix

Part 3 — Data Pipeline and Bigram Baseline
A (vocab, vocab) matrix of zeros to tally bigram counts into.
"""
import numpy as np  # noqa: F401


def allocate_count_matrix(vocab_size):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
