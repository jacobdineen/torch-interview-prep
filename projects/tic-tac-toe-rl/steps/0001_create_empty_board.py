"""
Step 0001: create_empty_board

Part 1 — Board Representation & Game Engine
A fresh 3x3 board of zeros (all cells empty).
"""
import numpy as np  # noqa: F401


def create_empty_board():
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
