"""
Step 0031: canonical_board_key

Part 3 — Tabular Q-Learning Foundations
Canonical key under the board's 8 symmetries (4 rotations x mirror): the
lexicographically smallest cell-tuple over all transforms. Shrinks the state space.
"""
import numpy as np  # noqa: F401


def canonical_board_key(board):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
