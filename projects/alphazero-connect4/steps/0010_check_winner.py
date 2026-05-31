"""
Step 0010: check_winner

Part 1 — Connect-4 Game Engine
Return the winning player (+1 / -1), or 0 if there is no winner.
"""
import torch  # noqa: F401


def check_winner(board):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
