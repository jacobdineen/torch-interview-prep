"""
Step 0059: evaluate_q_agent_vs_minimax

Part 4 — Self-Play, Evaluation & Persistence
Greedy agent (X) vs optimal minimax (O); return stats from X's view.
Against optimal play the agent can never win (best case is a draw).
"""
import numpy as np  # noqa: F401


def evaluate_q_agent_vs_minimax(q_table, n_games):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
