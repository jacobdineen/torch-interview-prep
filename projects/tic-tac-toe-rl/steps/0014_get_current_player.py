"""
Step 0014: get_current_player

Part 1 — Board Representation & Game Engine
Whose turn it is: +1 if X and O have played equally (X starts), else -1.
"""
import numpy as np  # noqa: F401


def get_current_player(board):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
