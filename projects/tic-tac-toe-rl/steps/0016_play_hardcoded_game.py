"""
Step 0016: play_hardcoded_game

Part 1 — Board Representation & Game Engine
Play a fixed list of actions, alternating from X, and return the final board.
"""
import numpy as np  # noqa: F401


def play_hardcoded_game(moves):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
