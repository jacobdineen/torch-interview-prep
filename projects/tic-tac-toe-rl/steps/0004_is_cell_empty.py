"""
Step 0004: is_cell_empty

Part 1 — Board Representation & Game Engine
True if the cell at action 0..8 is empty.
"""
import numpy as np  # noqa: F401


def is_cell_empty(board, action):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
