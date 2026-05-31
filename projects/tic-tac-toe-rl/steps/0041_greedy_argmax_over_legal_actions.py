"""
Step 0041: greedy_argmax_over_legal_actions

Part 3 — Tabular Q-Learning Foundations
The legal action with the highest Q-value (first if tied).
"""
import numpy as np  # noqa: F401


def greedy_argmax_over_legal_actions(q_table, board):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
