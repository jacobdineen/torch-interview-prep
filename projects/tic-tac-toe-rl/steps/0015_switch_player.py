"""
Step 0015: switch_player

Part 1 — Board Representation & Game Engine
The other player: +1 <-> -1.
"""
import numpy as np  # noqa: F401


def switch_player(player):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
