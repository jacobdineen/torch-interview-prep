"""
Step 0043: tic_tac_toe_reward

Part 3 — Tabular Q-Learning Foundations
Reward for ``player`` given a game status: +1 win, -1 loss, 0 otherwise.
"""
import numpy as np  # noqa: F401


def tic_tac_toe_reward(status, player):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
