"""
Step 0002: column_top_row

Part 1 — Connect-4 Game Engine
Row index a dropped piece would land in (lowest empty row), or -1 if full.
"""
import torch  # noqa: F401


def column_top_row(board, col):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
