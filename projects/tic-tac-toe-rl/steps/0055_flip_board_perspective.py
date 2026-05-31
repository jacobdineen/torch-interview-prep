"""
Step 0055: flip_board_perspective

Part 4 — Self-Play, Evaluation & Persistence
View the board from ``player``'s side: multiply by player so the mover's
pieces are always +1. One Q-table can then serve both sides.
"""
import numpy as np  # noqa: F401


def flip_board_perspective(board, player):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
