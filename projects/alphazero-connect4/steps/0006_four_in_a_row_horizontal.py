"""
Step 0006: four_in_a_row_horizontal

Part 1 — Connect-4 Game Engine
True if ``player`` has four consecutive pieces in any row.
"""
import torch  # noqa: F401


def four_in_a_row_horizontal(board, player):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
