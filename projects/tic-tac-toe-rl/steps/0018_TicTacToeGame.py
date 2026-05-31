"""
Step 0018: TicTacToeGame

Part 1 — Board Representation & Game Engine
A reusable game object wrapping the functional engine.

Attributes: ``board`` (3x3) and ``current_player`` (+1/-1).
Methods: reset(), legal_moves(), step(action) -> status, status().
"""
import numpy as np  # noqa: F401


class TicTacToeGame:
    def __init__(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def legal_moves(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def status(self):
        raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
