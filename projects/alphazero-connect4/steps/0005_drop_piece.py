"""
Step 0005: drop_piece

Part 1 — Connect-4 Game Engine
Return a NEW board with ``player``'s piece dropped into ``col``.
"""
import torch  # noqa: F401


def drop_piece(board, col, player):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
