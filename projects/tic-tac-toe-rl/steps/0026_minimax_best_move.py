"""
Step 0026: minimax_best_move

Part 2 — Random and Minimax Baselines
The action with the optimal minimax value for ``player`` (first if tied).
Scores all candidate moves with one shared transposition table.
"""
import numpy as np  # noqa: F401


def minimax_best_move(board, player):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
