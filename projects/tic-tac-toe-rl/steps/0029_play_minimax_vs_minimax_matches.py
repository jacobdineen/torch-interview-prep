"""
Step 0029: play_minimax_vs_minimax_matches

Part 2 — Random and Minimax Baselines
Minimax vs minimax — always a draw with optimal play; returns statuses.
"""
import numpy as np  # noqa: F401


def play_minimax_vs_minimax_matches(n_games):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
