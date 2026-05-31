"""
Step 0028: play_minimax_vs_random_matches

Part 2 — Random and Minimax Baselines
Minimax (X) vs random (O). Minimax should never lose; returns statuses.
Move selection uses alpha-beta (same as minimax, just faster).
"""
import numpy as np  # noqa: F401


def play_minimax_vs_random_matches(n_games, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
