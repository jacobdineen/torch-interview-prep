"""
Step 0019: random_move_agent

Part 2 — Random and Minimax Baselines
Pick a uniformly random legal action using ``rng`` (a numpy Generator).
"""
import numpy as np  # noqa: F401


def random_move_agent(board, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
