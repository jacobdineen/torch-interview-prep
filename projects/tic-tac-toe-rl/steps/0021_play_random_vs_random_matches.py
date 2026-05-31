"""
Step 0021: play_random_vs_random_matches

Part 2 — Random and Minimax Baselines
Play ``n_games`` random-vs-random games; return the list of statuses.
"""
import numpy as np  # noqa: F401


def play_random_vs_random_matches(n_games, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
