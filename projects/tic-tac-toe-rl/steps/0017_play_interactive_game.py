"""
Step 0017: play_interactive_game

Part 1 — Board Representation & Game Engine
Play to termination, asking ``get_move(board, player)`` for each action.
Returns the final game status (+1 / -1 / 0). ``get_move`` makes this testable
without real stdin (pass a scripted callback).
"""
import numpy as np  # noqa: F401


def play_interactive_game(get_move):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
