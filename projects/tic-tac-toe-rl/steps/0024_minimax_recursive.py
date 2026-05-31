"""
Step 0024: minimax_recursive

Part 2 — Random and Minimax Baselines
Exact minimax value of ``board`` with ``player`` to move (X maximizes). Uses
a transposition table (positions reachable many ways are scored once).
"""
import numpy as np  # noqa: F401


def minimax_recursive(board, player):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
