"""
Step 0039: epsilon_greedy_explore_move

Part 3 — Tabular Q-Learning Foundations
The explore branch: a uniformly random legal move.
"""
import numpy as np  # noqa: F401


def epsilon_greedy_explore_move(board, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
