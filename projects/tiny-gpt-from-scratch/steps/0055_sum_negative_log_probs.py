"""
Step 0055: sum_negative_log_probs

Part 3 — Data Pipeline and Bigram Baseline
Total negative log-likelihood of every adjacent pair in ``data``.
"""
import numpy as np  # noqa: F401


def sum_negative_log_probs(probs, data):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
