"""
Step 0023: minimax_terminal_score

Part 2 — Random and Minimax Baselines
Score of a terminal board from X's perspective: +1 X win, -1 O win, 0 draw.
"""
import numpy as np  # noqa: F401


def minimax_terminal_score(board):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
