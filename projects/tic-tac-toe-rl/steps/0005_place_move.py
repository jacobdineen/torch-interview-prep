"""
Step 0005: place_move

Part 1 — Board Representation & Game Engine
Return a NEW board with ``player``'s mark placed at ``action`` (no mutation).
"""
import numpy as np  # noqa: F401


def place_move(board, action, player):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
