"""
Step 0003: print_board

Part 1 — Board Representation & Game Engine
Render the board as a string using X / O / . , rows on separate lines.
"""
import numpy as np  # noqa: F401


def print_board(board):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
