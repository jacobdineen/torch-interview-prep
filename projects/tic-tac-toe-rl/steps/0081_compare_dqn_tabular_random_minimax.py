"""
Step 0081: compare_dqn_tabular_random_minimax

Part 5 — Deep Q-Network Agent
Win/loss/draw vs a random opponent for each agent (DQN, tabular, random).
Returns a dict keyed by agent name.
"""
import numpy as np  # noqa: F401


def compare_dqn_tabular_random_minimax(dqn_params, q_table, n_games, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
