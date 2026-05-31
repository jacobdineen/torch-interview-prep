"""
Step 0030: encode_board_state_key

Part 3 — Tabular Q-Learning Foundations
A hashable key for a board: the tuple of its 9 cells (row-major).
"""
import numpy as np  # noqa: F401


def encode_board_state_key(board):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
