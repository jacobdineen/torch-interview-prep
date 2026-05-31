"""
Step 0002: encode_player

Part 1 — Board Representation & Game Engine
Map a player symbol to its board mark: 'X' -> +1, 'O' -> -1.
"""
import numpy as np  # noqa: F401


def encode_player(symbol):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
