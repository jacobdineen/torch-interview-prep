"""
Step 0048: episode_agent_pick_action

Part 3 — Tabular Q-Learning Foundations
The agent's action for this step (epsilon-greedy).
"""
import numpy as np  # noqa: F401


def episode_agent_pick_action(q_table, board, epsilon, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
