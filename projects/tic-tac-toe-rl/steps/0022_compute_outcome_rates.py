"""
Step 0022: compute_outcome_rates

Part 2 — Random and Minimax Baselines
Fraction of X wins / O wins / draws from a list of statuses.
"""
import numpy as np  # noqa: F401


def compute_outcome_rates(statuses):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
