"""
Step 0020: play_random_vs_random_game

Part 2 — Random and Minimax Baselines
Play one game with both sides random; return the final status (+1/-1/0).
"""
import numpy as np  # noqa: F401


def play_random_vs_random_game(rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
