"""
Step 0012: is_terminal

Part 1 — Connect-4 Game Engine
True if the game is over (someone won or the board is full).
"""
import torch  # noqa: F401


def is_terminal(board):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
