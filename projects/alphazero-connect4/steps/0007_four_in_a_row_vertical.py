"""
Step 0007: four_in_a_row_vertical

Part 1 — Connect-4 Game Engine
True if ``player`` has four consecutive pieces in any column.
"""
import torch  # noqa: F401


def four_in_a_row_vertical(board, player):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
